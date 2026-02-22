package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/oleg-safonov/nlp"
	nlprudata "github.com/oleg-safonov/nlp-ru-data"
)

type Token struct {
	Form     string
	Expected string
	UPOS     string
}

type ErrorKey struct {
	Word string
	Pred string
	Exp  string
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <file.conllu>")
		return
	}

	filePath := os.Args[1]

	base, err := nlprudata.Load()
	if err != nil {
		log.Fatal(err)
	}

	lem, err := nlp.NewLemmatizer(base)
	if err != nil {
		log.Fatal(err)
	}

	correct, processedCount, errorCounts, duration := test(filePath, lem)

	accuracy := float64(correct) / float64(processedCount) * 100
	speed := float64(processedCount) / duration.Seconds()

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("РЕЗУЛЬТАТЫ Go Lib:\n")
	fmt.Printf("Точность (Accuracy):  %.2f%%\n", accuracy)
	fmt.Printf("Скорость:             %.2f ток/сек\n", speed)
	fmt.Println(strings.Repeat("=", 60))

	type errorStat struct {
		Key   ErrorKey
		Count int
	}
	var stats []errorStat
	for k, v := range errorCounts {
		stats = append(stats, errorStat{k, v})
	}
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].Count > stats[j].Count
	})

	fmt.Printf("\nТОП-20 ОШИБОК:\n")
	fmt.Printf("%-15s | %-15s | %-15s | %s\n", "Слово", "Go Lib", "Ожидалось", "Кол-во")
	fmt.Println(strings.Repeat("-", 70))
	for i := 0; i < 20 && i < len(stats); i++ {
		s := stats[i]
		fmt.Printf("%-15s | %-15s | %-15s | %d\n", s.Key.Word, s.Key.Pred, s.Key.Exp, s.Count)
	}
}

func test(filePath string, lem *nlp.Lemmatizer) (int, int, map[ErrorKey]int, time.Duration) {
	excludedPos := map[string]bool{"PUNCT": true, "_": true, "X": true, "H": true}

	sentences := readTestFile(filePath, excludedPos)

	correct := 0
	processedCount := 0
	errorCounts := make(map[ErrorKey]int)

	startTime := time.Now()

	finalReplace := map[string]string{
		"об": "о", "обо": "о",
		"со":  "с",
		"во":  "в",
		"тот": "то",
		"ко":  "к",
	}
	for _, sent := range sentences {
		sentence := make([]string, len(sent))
		for i := range sent {
			sentence[i] = sent[i].Form
		}
		tokens := nlp.CreateTokens(sentence)
		results := lem.LemmatizeTokens(tokens)

		for i, token := range sent {
			if excludedPos[token.UPOS] {
				continue
			}

			predicted := strings.ToLower(results[i])
			if _, ok := finalReplace[predicted]; ok {
				predicted = finalReplace[predicted]
			}

			expected := nlp.Normalize(token.Expected)
			if _, ok := finalReplace[expected]; ok {
				expected = finalReplace[expected]
			}

			processedCount++
			if predicted == expected {
				correct++
			} else {
				//fmt.Println(strings.ToLower(token.Form), predicted, token.Expected)
				errorCounts[ErrorKey{
					Word: strings.ToLower(token.Form),
					Pred: predicted,
					Exp:  token.Expected,
				}]++
			}
		}
	}

	duration := time.Since(startTime)

	return correct, processedCount, errorCounts, duration
}

func readTestFile(filePath string, excludedPos map[string]bool) [][]Token {
	fmt.Printf("--- Загрузка данных из %s ---\n", filePath)

	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Ошибка открытия файла: %v", err)
	}
	defer file.Close()

	var sentences [][]Token
	var currentSent []Token

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			if len(currentSent) > 0 {
				sentences = append(sentences, currentSent)
				currentSent = nil
			}
			continue
		}

		if strings.HasPrefix(line, "#") {
			continue
		}

		fields := strings.Split(line, "\t")
		if len(fields) < 4 {
			continue
		}
		if excludedPos[fields[3]] {
			continue
		}
		currentSent = append(currentSent, Token{
			Form:     fields[1],
			Expected: strings.ToLower(fields[2]),
			UPOS:     fields[3],
		})
	}

	if len(currentSent) > 0 {
		sentences = append(sentences, currentSent)
	}

	totalTokens := 0
	for _, s := range sentences {
		totalTokens += len(s)
	}

	fmt.Printf("Загружено предложений: %d (всего токенов: %d)\n", len(sentences), totalTokens)
	fmt.Println("--- Запуск тестирования ---")
	return sentences
}
