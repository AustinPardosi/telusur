"""Query embedder sidecar for `telusur serve --embedder http://127.0.0.1:8001`.

    uv run --with fastembed scripts/embed_server.py --model nomic-ai/nomic-embed-text-v1.5

POST /embed {"text": "..."} returns {"vector": [...]}: the query embedded the way
prep_beir.py embeds queries (same model, query prefix and layer norm), so it matches the
corpus vectors. One request at a time; --port defaults to 8001.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from prep_beir import PREFIXES, layer_norm


def main(args: list[str]) -> None:
    opt = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    from fastembed import TextEmbedding

    model_name = opt("--model", "nomic-ai/nomic-embed-text-v1.5")
    prefix = PREFIXES.get(model_name, ("", ""))[1]
    model = TextEmbedding(model_name)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/embed":
                return self.send_error(404)
            try:
                text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
            except (ValueError, KeyError, TypeError):
                return self.send_error(400, 'expected {"text": "..."}')
            row = layer_norm(next(iter(model.embed([prefix + text])))[None, :])[0]
            body = json.dumps({"vector": row.tolist()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    port = int(opt("--port", "8001"))
    print(f"embedding with {model_name} on http://127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
