package main

import (
	"log"
	"testing"

	"github.com/oleg-safonov/nlp"
	nlprudata "github.com/oleg-safonov/nlp-ru-data"
)

func BenchmarkTokenizer(b *testing.B) {
	base, err := nlprudata.Load()
	if err != nil {
		log.Fatal(err)
	}

	lem, err := nlp.NewLemmatizer(base)
	if err != nil {
		log.Fatal(err)
	}

	b.ResetTimer()
	test("data/ru_syntagrus-ud-test.conllu", lem)
}
