import unittest
from pathlib import Path
from reproduce import read_jsonl

class MultilingualDatasetTest(unittest.TestCase):
    def test_paired_dataset(self):
        rows = read_jsonl(Path(__file__).resolve().parents[1] / 'multilingual_jbb' / 'translations.jsonl')
        self.assertEqual(len(rows), 236)
        index = {(r['id'], r['language']): r for r in rows}
        self.assertEqual(len(index), 236)
        for i in range(59):
            ident = f'jbb-{i:04d}'
            english = index[(ident, 'en')]
            self.assertEqual(english['goal'], english['prompt'])
            for language in ('en', 'zh-CN', 'ja', 'ar'):
                row = index[(ident, language)]
                self.assertEqual(row['goal'], english['goal'])
                self.assertTrue(row['prompt'].strip())

if __name__ == '__main__': unittest.main()
