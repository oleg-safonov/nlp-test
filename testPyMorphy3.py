import time
import argparse
from collections import Counter
from conllu import parse_incr
import pymorphy3

def normalize_lemma(lemma):
    # приводим к одному виду предлоги
    mapping = {
        'об': 'о', 'обо': 'о',
        'со': 'с',
        'во': 'в',
        'тот':'то',
        'ко': 'к'
    }
    lemma = lemma.replace('ё', 'е').replace('.', '')
    return mapping.get(lemma, lemma)

def test_pymorphy_errors(file_path, top_n=20):
    morph = pymorphy3.MorphAnalyzer(result_type=None)
    excluded_pos = {'PUNCT', '_', 'X', 'H'}
    
    lemma_cache = {}
    
    test_cases = []
    print(f"--- Загрузка данных из {file_path} ---")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence_data = []
            for token in tokenlist:
                if token['upos'] not in excluded_pos:
                    sentence_data.append({
                        'word': token['form'],
                        'expected': token['lemma'].lower() if token['lemma'] else ""
                    })
            if sentence_data:
                test_cases.append(sentence_data)

    total_tokens = 0
    correct_predictions = 0
    errors = Counter()
    
    print(f"Токенов для проверки: {sum(len(s) for s in test_cases)}")
    print("--- Запуск тестирования ---")

    start_time = time.perf_counter()
    
    for sentence in test_cases:
        for case in sentence:
            total_tokens += 1
            word = case['word']
            word_lower = word.lower()
            
            # Работа с кэшем
            if word_lower in lemma_cache:
                predicted_lemma = lemma_cache[word_lower]
            else:
                parsed = morph.parse(word_lower)
                predicted_lemma = parsed[0][2] if parsed else word_lower
                lemma_cache[word_lower] = predicted_lemma
            
            if normalize_lemma(predicted_lemma) == normalize_lemma(case['expected']):
                correct_predictions += 1
            else:
                error_key = (word_lower, predicted_lemma, case['expected'])
                errors[error_key] += 1
                
    end_time = time.perf_counter()
    
    # Статистика
    accuracy = (correct_predictions / total_tokens) * 100 if total_tokens > 0 else 0
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТЫ PYMORPHY3:")
    print(f"Точность (Accuracy):  {accuracy:.2f}%")
    print(f"Скорость:             {total_tokens / total_time:.2f} ток/сек")
    print("="*60)
    
    print(f"\nТОП-{top_n} ЧАСТЫХ ОШИБОК:")
    print(f"{'Слово':<15} | {'PyMorphy3':<15} | {'Ожидалось':<15} | {'Кол-во'}")
    print("-" * 70)
    
    for (word, pred, exp), count in errors.most_common(top_n):
        print(f"{word:<15} | {pred:<15} | {exp:<15} | {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    test_pymorphy_errors(args.file)
