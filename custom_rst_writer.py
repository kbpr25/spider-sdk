from astropy.io.ascii import RST

class CustomRST(RST):
    def __init__(self, header_rows=None, **kwargs):
        super().__init__(**kwargs)
        self.header_rows = header_rows

    def _write_lines(self, lines):
        if self.header_rows:
            # Add header rows before the table
            header_line = ' '.join(self.header_rows)
            lines.insert(0, '=' * len(header_line))
            lines.insert(0, header_line)
            lines.insert(0, '=' * len(header_line))
        super()._write_lines(lines)