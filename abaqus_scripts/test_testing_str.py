import unittest
import os
from unittest.mock import patch

import testing_str


class TestTestingStr(unittest.TestCase):
    def test_get_files_from_directory(self):
        with patch('os.listdir') as mock_listdir, patch('os.path.isfile') as mock_isfile:
            mock_listdir.return_value = ['a.odb', 'b.txt', 'c']
            mock_isfile.side_effect = lambda x: not x.endswith('c')
            files = testing_str.get_files_from_directory('dummy_dir')
            self.assertEqual(files, ['a.odb', 'b.txt'])

    def test_get_files_with_max_fric(self):
        files = [
            'foo_fric_0100.odb',
            'foo_fric_025.odb',
            'bar_fric_050.odb',
            'bar_fric_0100.odb',
            'baz.odb'
        ]
        result = testing_str.get_files_with_max_fric(files)
        self.assertEqual(result['foo'], 'foo_fric_0100.odb')
        self.assertEqual(result['bar'], 'bar_fric_0100.odb')
        self.assertNotIn('baz', result)
        self.assertEqual(len(result), 2)

    def test_get_missing_files(self):
        input_files = [
            'foo_fric_0100.odb',
            'bar_fric_025.odb',
            'baz_fric_050.odb'
        ]
        pic_files = [
            'foo_fric_0100.png'
        ]
        with patch('testing_str.get_files_from_directory', side_effect=[input_files, pic_files]):
            with patch('testing_str.get_files_with_max_fric', return_value={'foo': 'foo_fric_0100.odb', 'bar': 'bar_fric_025.odb', 'baz': 'baz_fric_050.odb'}):
                missing = testing_str.get_missing_files('input_dir', 'pic_dir')
                self.assertIn('bar_fric_025.odb', missing)
                self.assertIn('baz_fric_050.odb', missing)
                self.assertNotIn('foo_fric_0100.odb', missing)

    def test_main_prints(self):
        # Patch all filesystem and print
        with patch('testing_str.get_missing_files', return_value=['foo.odb']), \
             patch('builtins.print') as mock_print:
            testing_str.main()
            mock_print.assert_any_call("Missing files:")
            mock_print.assert_any_call('foo.odb')


if __name__ == '__main__':
    unittest.main()
