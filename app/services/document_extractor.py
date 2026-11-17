"""Offline, bounded text extraction for the established supported document formats."""
import io
import re
import zipfile
import xml.etree.ElementTree as element_tree
from pathlib import Path


class DocumentExtractor:
    """Extract text without retaining files or adding document-processing dependencies."""

    def __init__(self, max_archive_members, max_archive_uncompressed_bytes):
        self.max_archive_members = max_archive_members
        self.max_archive_uncompressed_bytes = max_archive_uncompressed_bytes

    def extract(self, name, content):
        return '\n'.join(segment['text'] for segment in self.extract_with_provenance(name, content))

    def extract_with_provenance(self, name, content):
        """Return text segments with only source locations reliably available offline."""
        extension = Path(name).suffix.lower()
        if extension in ('.txt', '.md', '.csv', '.json', '.log'):
            return [{'text': self._text(content)}]
        if extension == '.docx':
            return self._docx_segments(content)
        if extension == '.xlsx':
            return self._xlsx_segments(content)
        if extension == '.pdf':
            return self._pdf_segments(content)
        raise ValueError('فرمت فایل پشتیبانی نمی‌شود')

    def validate_archive(self, archive):
        """Reject oversized OOXML ZIPs before XML is read into memory."""
        infos = archive.infolist()
        if (len(infos) > self.max_archive_members or
                sum(info.file_size for info in infos) > self.max_archive_uncompressed_bytes):
            raise ValueError('محتوای فشرده سند بیش از حد بزرگ است')

    @staticmethod
    def _text(content):
        try:
            return content.decode('utf-8-sig')
        except UnicodeError:
            pass
        # UTF-16 has a BOM or null-byte pattern; trying it on arbitrary legacy
        # single-byte text can silently produce unrelated Unicode characters.
        if content.startswith((b'\xff\xfe', b'\xfe\xff')) or b'\x00' in content[:128]:
            try:
                return content.decode('utf-16')
            except UnicodeError:
                pass
        try:
            return content.decode('cp1256')
        except UnicodeError:
            pass
        return content.decode('utf-8', errors='replace')

    def _docx_segments(self, content):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.validate_archive(archive)
            if ('word/document.xml' not in archive.namelist() or
                    archive.getinfo('word/document.xml').file_size > 20_000_000):
                raise ValueError('سند Word بیش از حد بزرگ است')
            root = element_tree.fromstring(archive.read('word/document.xml'))
        namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        heading = ''
        segments = []
        for paragraph in root.iter(namespace + 'p'):
            text = ''.join(item.text or '' for item in paragraph.iter(namespace + 't'))
            style = paragraph.find('.//' + namespace + 'pStyle')
            style_value = style.get(namespace + 'val', '') if style is not None else ''
            if style_value.lower().startswith('heading') and text.strip():
                heading = text.strip()
            if text:
                item = {'text': text}
                if heading:
                    item['heading'] = heading
                segments.append(item)
        return segments

    def _xlsx_segments(self, content):
        namespace = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.validate_archive(archive)
            shared = []
            if 'xl/sharedStrings.xml' in archive.namelist():
                if archive.getinfo('xl/sharedStrings.xml').file_size > 20_000_000:
                    raise ValueError('فایل Excel بیش از حد بزرگ است')
                root = element_tree.fromstring(archive.read('xl/sharedStrings.xml'))
                shared = [''.join(text.text or '' for text in item.iter(namespace + 't'))
                          for item in root.iter(namespace + 'si')]
            sheet_titles = {}
            if 'xl/workbook.xml' in archive.namelist():
                workbook = element_tree.fromstring(archive.read('xl/workbook.xml'))
                for index, sheet in enumerate(workbook.iter(namespace + 'sheet'), 1):
                    sheet_titles[index] = sheet.get('name', 'Sheet' + str(index))
            segments = []
            sheet_names = sorted(name for name in archive.namelist()
                                 if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name))
            for sheet_name in sheet_names:
                if archive.getinfo(sheet_name).file_size > 20_000_000:
                    raise ValueError('برگه Excel بیش از حد بزرگ است')
                root = element_tree.fromstring(archive.read(sheet_name))
                worksheet = sheet_titles.get(int(re.search(r'(\d+)', sheet_name).group(1)), sheet_name)
                for row in root.iter(namespace + 'row'):
                    cells = []
                    for cell in row.iter(namespace + 'c'):
                        value_node = cell.find(namespace + 'v')
                        value = value_node.text if value_node is not None and value_node.text else ''
                        if cell.get('t') == 's' and value.isdigit() and int(value) < len(shared):
                            value = shared[int(value)]
                        if cell.get('t') == 'inlineStr':
                            value = ''.join(text.text or '' for text in cell.iter(namespace + 't'))
                        if value:
                            cells.append(cell.get('r', '') + '=' + value)
                    if cells:
                        number = int(row.get('r', '0') or 0) or None
                        segments.append({'text': '[' + worksheet + '] ' + ' | '.join(cells),
                                         'worksheet': worksheet, 'row_start': number, 'row_end': number})
            return segments

    @staticmethod
    def _pdf_segments(content):
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError('برای PDF متنی کتابخانه pypdf را آفلاین نصب کنید') from error
        return [{'text': page.extract_text() or '', 'page_start': number, 'page_end': number}
                for number, page in enumerate(PdfReader(io.BytesIO(content)).pages, 1)
                if page.extract_text()]
