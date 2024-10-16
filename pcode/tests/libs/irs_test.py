import unittest
from datetime import datetime, timedelta
class TestGetIMMCal(unittest.TestCase):
    def test_imm_dates(self):
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2026, 12, 31)
        expected_imm_dates = [
            datetime(2024, 3, 20),
            datetime(2024, 6, 19),
            datetime(2024, 9, 18),
            datetime(2024, 12, 18),
            datetime(2025, 3, 19),
            datetime(2025, 6, 18),
            datetime(2025, 9, 17),
            datetime(2025, 12, 17),
            datetime(2026, 3, 18),
            datetime(2026, 6, 17),
            datetime(2026, 9, 16),
            datetime(2026, 12, 16)
        ]
        result = getIMMCal(start_date, end_date)
        self.assertEqual(result, expected_imm_dates)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)