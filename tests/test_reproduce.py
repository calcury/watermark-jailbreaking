import unittest
from reproduce import parse_judge, validate_resume


class JudgeParsingTests(unittest.TestCase):
    def test_resume_checks_source_and_configuration(self):
        row = {'id': '1', 'response': 'text'}
        saved = {**row, 'wildguard': 0, 'judge_model': 'model', 'judge_quantization': '4bit'}
        validate_resume([row], [saved], 'wildguard', 'model', '4bit')
        with self.assertRaises(ValueError):
            validate_resume([row], [saved], 'wildguard', 'model', 'none')
        with self.assertRaises(ValueError):
            validate_resume([{**row, 'response': 'changed'}], [saved], 'wildguard', 'model', '4bit')

    def test_wildguard_labels(self):
        self.assertEqual(parse_judge('wildguard', 'Harmful response: yes'), 1)
        self.assertEqual(parse_judge('wildguard', 'Harmful response: no'), 0)
        with self.assertRaises(ValueError):
            parse_judge('wildguard', 'Harmful request: yes')


if __name__ == '__main__':
    unittest.main()
