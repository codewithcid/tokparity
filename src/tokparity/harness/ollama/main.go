// Tokenizes text with Ollama's Go tokenizer (mlxrunner/tokenizer), the way Ollama loads it.
//
// usage: ollama-encode TOKENIZER_DIR
// Reads one JSON array of strings per line on stdin and writes one JSON array of token id arrays per line
// on stdout, so the tokenizer is loaded once per run.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func read(dir, name string) []byte {
	b, _ := os.ReadFile(filepath.Join(dir, name))
	return b
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: ollama-encode TOKENIZER_DIR")
		os.Exit(2)
	}
	dir := os.Args[1]
	tok, err := tokenizer.LoadFromBytesWithConfig(read(dir, "tokenizer.json"), &tokenizer.TokenizerConfig{
		TokenizerConfigJSON:  read(dir, "tokenizer_config.json"),
		GenerationConfigJSON: read(dir, "generation_config.json"),
		SpecialTokensMapJSON: read(dir, "special_tokens_map.json"),
		ConfigJSON:           read(dir, "config.json"),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(2)
	}

	in := bufio.NewReaderSize(os.Stdin, 1<<20)
	out := bufio.NewWriter(os.Stdout)
	for {
		line, err := in.ReadBytes('\n')
		if len(line) > 0 {
			var texts []string
			if err := json.Unmarshal(line, &texts); err != nil {
				fmt.Fprintln(os.Stderr, "bad request:", err)
				os.Exit(2)
			}
			ids := make([][]int32, len(texts))
			for i, t := range texts {
				ids[i] = tok.Encode(t, false)
				if ids[i] == nil {
					ids[i] = []int32{}
				}
			}
			b, _ := json.Marshal(ids)
			out.Write(b)
			out.WriteByte('\n')
			out.Flush()
		}
		if err == io.EOF {
			return
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "read:", err)
			os.Exit(2)
		}
	}
}
