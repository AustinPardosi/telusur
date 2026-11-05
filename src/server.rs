//! HTTP API: `POST /search`, `GET /health`, `GET /stats`. Query text is embedded by a sidecar
//! (`scripts/embed_server.py`, the model the corpus was embedded with) unless the request
//! brings its own vector. When embedding fails, hybrid falls back to keyword and says why;
//! an explicit neural request gets a 503.

use crate::{
    engine::{FallbackReason, Index, MAX_K, Mode, Request},
    filter::Filters,
    vector::kernel::Kernel,
};
use anyhow::Result;
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

/// How long a search waits for the embedder.
pub const EMBED_TIMEOUT: Duration = Duration::from_secs(1);

struct App {
    index: Index,
    dir: PathBuf,
    /// Base URL; the server posts `{"text": ...}` to `<embedder>/embed` and expects
    /// `{"vector": [...]}` back.
    embedder: Option<String>,
    client: reqwest::Client,
}

/// The routes, serving the index in `dir`.
pub fn router(dir: &Path, embedder: Option<String>) -> Result<Router> {
    let app = App {
        index: Index::open(dir)?,
        dir: dir.to_path_buf(),
        embedder: embedder.map(|u| u.trim_end_matches('/').to_string()),
        client: reqwest::Client::builder().timeout(EMBED_TIMEOUT).build()?,
    };
    Ok(Router::new()
        .route("/search", post(search))
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/stats", get(stats))
        .with_state(Arc::new(app)))
}

pub async fn serve(dir: &Path, addr: &str, embedder: Option<String>) -> Result<()> {
    let app = router(dir, embedder)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("serving {} on http://{addr}", dir.display());
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchBody {
    #[serde(default)]
    query: String,
    #[serde(default)]
    mode: Mode,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filters: Filters,
    vector: Option<Vec<f32>>,
    #[serde(default)]
    debug: bool,
}

fn default_k() -> usize {
    10
}

fn error(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({ "error": msg.to_string() }))).into_response()
}

async fn search(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let start = Instant::now();
    let body: SearchBody = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if body.k > MAX_K {
        return error(
            StatusCode::BAD_REQUEST,
            format!("k must be at most {MAX_K}, got {}", body.k),
        );
    }
    let mut req = Request {
        query: body.query,
        mode: body.mode,
        k: body.k,
        debug: body.debug,
        vector: body.vector,
        filters: body.filters,
        ..Request::default()
    };

    let (mut embed_ms, mut fallback) = (None, None);
    if req.vector.is_none()
        && req.mode != Mode::Keyword
        && let Some(dims) = app.index.vector_dims()
    {
        let t = Instant::now();
        match app.embed(&req.query, dims).await {
            Ok(v) => req.vector = Some(v),
            Err((_, msg)) if req.mode == Mode::Neural => {
                return error(StatusCode::SERVICE_UNAVAILABLE, msg);
            }
            Err((reason, _)) => (req.mode, fallback) = (Mode::Keyword, Some(reason)),
        }
        if app.embedder.is_some() {
            embed_ms = Some(t.elapsed().as_secs_f64() * 1e3);
        }
    }

    let app2 = Arc::clone(&app);
    match tokio::task::spawn_blocking(move || app2.index.search(&req)).await {
        Ok(Ok(mut res)) => {
            res.fallback_reason = res.fallback_reason.or(fallback);
            res.timings_ms.embed = embed_ms;
            res.timings_ms.total = start.elapsed().as_secs_f64() * 1e3;
            Json(res).into_response()
        }
        // ponytail: every engine error is a bad request (filters, dims, no vectors); split out
        // internal errors if a served index can ever fail to read.
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

impl App {
    /// The query vector for `text` from the embedder, or why there is none.
    async fn embed(&self, text: &str, dims: usize) -> Result<Vec<f32>, (FallbackReason, String)> {
        #[derive(Deserialize)]
        struct Embedded {
            vector: Vec<f32>,
        }
        let Some(url) = &self.embedder else {
            return Err((
                FallbackReason::NoEmbedder,
                "neural search needs a query vector, and no embedder is configured".into(),
            ));
        };
        let failed = |e: reqwest::Error| match e.is_timeout() {
            true => (
                FallbackReason::EmbedderTimeout,
                format!("the embedder did not answer within {EMBED_TIMEOUT:?}"),
            ),
            false => (FallbackReason::EmbedderError, format!("embedder: {e}")),
        };
        let res = self
            .client
            .post(format!("{url}/embed"))
            .json(&json!({ "text": text }))
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(failed)?;
        let v = res.json::<Embedded>().await.map_err(failed)?.vector;
        if v.len() != dims {
            return Err((
                FallbackReason::EmbedderError,
                format!("the embedder sent {} dims; the index has {dims}", v.len()),
            ));
        }
        Ok(v)
    }
}

async fn stats(State(app): State<Arc<App>>) -> Response {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(&app.dir).into_iter().flatten().flatten() {
        if let Ok(m) = entry.metadata() {
            files.insert(entry.file_name().to_string_lossy().into_owned(), m.len());
        }
    }
    let kernel = Kernel::default_for_process().map_or(Value::Null, |k| json!(k));
    Json(json!({
        "docs": app.index.docs(),
        "vector_dims": app.index.vector_dims(),
        "kernel": kernel,
        "embedder": app.embedder,
        "index_bytes": files.values().sum::<u64>(),
        "files": files,
    }))
    .into_response()
}
