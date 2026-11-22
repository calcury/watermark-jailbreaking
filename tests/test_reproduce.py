import unittest
from reproduce import parse_judge


class JudgeParsingTests(unittest.TestCase):
    def test_guard_labels(self):
        self.assertEqual(parse_judge('llamaguard', 'unsafe\nS1'), 1)
        self.assertEqual(parse_judge('llamaguard', 'safe'), 0)
        with self.assertRaises(ValueError):
            parse_judge('llamaguard', 'unknown')

    def test_wildguard_labels(self):
        self.assertEqual(parse_judge('wildguard', 'Harmful response: yes'), 1)
        self.assertEqual(parse_judge('wildguard', 'Harmful response: no'), 0)
        with self.assertRaises(ValueError):
            parse_judge('wildguard', 'Harmful request: yes')


if __name__ == '__main__':
    unittest.main()
