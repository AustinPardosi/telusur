// Prints the tokens of Anserini's DefaultEnglishAnalyzer chain (Lucene 10.5.0, the version
// Anserini pins) for each stdin line, space-separated. A literal \n in a line becomes a newline.
// This is how the Lucene token streams in tests/analyzer.rs were captured:
//
//   java -cp lucene-core-10.5.0.jar:lucene-analysis-common-10.5.0.jar scripts/lucene_tokens.java < inputs.txt

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.en.*;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

public class lucene_tokens {
  public static void main(String[] args) throws IOException {
    Analyzer analyzer = new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer source = new StandardTokenizer();
        TokenStream ts = new EnglishPossessiveFilter(source);
        ts = new LowerCaseFilter(ts);
        ts = new StopFilter(ts, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
        ts = new PorterStemFilter(ts);
        return new TokenStreamComponents(source, ts);
      }
    };
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
    for (String line; (line = in.readLine()) != null; ) {
      List<String> tokens = new ArrayList<>();
      try (TokenStream ts = analyzer.tokenStream("f", line.replace("\\n", "\n"))) {
        CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
        ts.reset();
        while (ts.incrementToken()) tokens.add(term.toString());
        ts.end();
      }
      out.println(String.join(" ", tokens));
    }
  }
}
