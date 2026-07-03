"""测试 arXiv PDF 下载和验证功能"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from services.arxiv.arxiv_search_service import ArxivSearchService


def create_minimal_valid_pdf() -> bytes:
    """创建一个最小但完全有效的 PDF（pypdf 可解析）"""
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""


class TestPdfValidation:
    """测试 PDF 验证功能"""

    def test_valid_pdf(self, tmp_path):
        """测试验证有效的 PDF 文件"""
        service = ArxivSearchService()

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(create_minimal_valid_pdf())

        assert service._is_valid_pdf(str(pdf_path))

    def test_invalid_pdf_wrong_header(self, tmp_path):
        """测试检测错误的文件头"""
        service = ArxivSearchService()

        pdf_path = tmp_path / "fake.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b'<html><body>404 Not Found</body></html>')

        assert not service._is_valid_pdf(str(pdf_path))

    def test_invalid_pdf_too_small(self, tmp_path):
        """测试检测文件过小"""
        service = ArxivSearchService()

        pdf_path = tmp_path / "tiny.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b'%PDF-1.4\n')  # 只有几个字节

        assert not service._is_valid_pdf(str(pdf_path))

    def test_invalid_pdf_missing_eof(self, tmp_path):
        """测试检测缺少 EOF 标记"""
        service = ArxivSearchService()

        pdf_path = tmp_path / "incomplete.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'%' + b'\xE2\xE3\xCF\xD3' + b'\n')
            # 写入足够的内容但不包含 %%EOF
            f.write(b'1 0 obj\n<< /Type /Catalog >>\nendobj\n' * 100)

        assert not service._is_valid_pdf(str(pdf_path))

    def test_invalid_pdf_corrupted_structure(self, tmp_path):
        """测试检测 PDF 结构损坏（有魔数和 EOF，但无法解析）

        注意：此测试只在 pypdf 可用时生效，否则会降级为基础验证（通过）
        """
        service = ArxivSearchService()

        pdf_path = tmp_path / "corrupted.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'%' + b'\xE2\xE3\xCF\xD3' + b'\n')
            # 写入无效的 PDF 对象结构
            f.write(b'garbage data that looks like PDF but is not\n' * 100)
            f.write(b'%%EOF\n')

        # 检查 pypdf 是否可用
        try:
            import pypdf
            # pypdf 可用时应该能检测出结构损坏
            assert not service._is_valid_pdf(str(pdf_path))
        except ImportError:
            # pypdf 不可用时，降级为基础验证（只检查魔数和 EOF），会通过
            assert service._is_valid_pdf(str(pdf_path))

    def test_nonexistent_file(self):
        """测试处理不存在的文件"""
        service = ArxivSearchService()
        assert not service._is_valid_pdf("/path/that/does/not/exist.pdf")


class TestPdfDownload:
    """测试 PDF 下载功能"""

    @pytest.fixture
    def service(self, tmp_path):
        """创建一个使用临时目录的服务实例"""
        service = ArxivSearchService()
        service.papers_dir = str(tmp_path)
        return service

    def test_download_new_pdf_success(self, service, tmp_path):
        """测试成功下载新的 PDF"""
        mock_response = Mock()
        mock_response.content = create_minimal_valid_pdf()
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = Mock()

        with patch.object(service, '_make_request_with_retry', return_value=mock_response):
            filepath = service.download_pdf("https://arxiv.org/pdf/2301.00001.pdf", "2301.00001")

            assert os.path.exists(filepath)
            assert filepath.endswith("2301.00001.pdf")
            # 验证文件内容
            with open(filepath, "rb") as f:
                content = f.read()
                assert content.startswith(b'%PDF-1.4')
                assert b'%%EOF' in content

    def test_download_existing_valid_pdf(self, service, tmp_path):
        """测试文件已存在且有效时，直接返回"""
        pdf_path = tmp_path / "2301.00002.pdf"
        pdf_path.write_bytes(create_minimal_valid_pdf())

        # 不应该触发下载
        with patch.object(service, '_make_request_with_retry') as mock_request:
            filepath = service.download_pdf("https://arxiv.org/pdf/2301.00002.pdf", "2301.00002")

            assert filepath == str(pdf_path)
            mock_request.assert_not_called()

    def test_download_existing_invalid_pdf_redownload(self, service, tmp_path):
        """测试文件已存在但无效时，重新下载"""
        # 创建一个无效的 PDF
        pdf_path = tmp_path / "2301.00003.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b'<html>404</html>')  # 假的 PDF

        mock_response = Mock()
        mock_response.content = create_minimal_valid_pdf()
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = Mock()

        with patch.object(service, '_make_request_with_retry', return_value=mock_response):
            filepath = service.download_pdf("https://arxiv.org/pdf/2301.00003.pdf", "2301.00003")

            # 应该触发重新下载
            assert os.path.exists(filepath)
            with open(filepath, "rb") as f:
                content = f.read()
                assert content.startswith(b'%PDF-1.4')

    def test_download_invalid_content_type(self, service):
        """测试响应不是 PDF 类型时抛出异常"""
        mock_response = Mock()
        mock_response.content = b'<html>404 Not Found</html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status = Mock()

        with patch.object(service, '_make_request_with_retry', return_value=mock_response):
            with pytest.raises(ValueError, match="Response is not a PDF"):
                service.download_pdf("https://arxiv.org/pdf/2301.00004.pdf", "2301.00004")

    def test_download_invalid_pdf_content(self, service):
        """测试下载的内容不是有效 PDF 时抛出异常"""
        mock_response = Mock()
        mock_response.content = b'<html>Error page</html>'
        mock_response.headers = {'Content-Type': 'application/pdf'}  # 声称是 PDF 但实际不是
        mock_response.raise_for_status = Mock()

        with patch.object(service, '_make_request_with_retry', return_value=mock_response):
            with pytest.raises(ValueError, match="Downloaded file is not a valid PDF"):
                service.download_pdf("https://arxiv.org/pdf/2301.00005.pdf", "2301.00005")

    def test_download_network_error_cleanup(self, service, tmp_path):
        """测试网络错误时清理临时文件"""
        import requests

        with patch.object(service, '_make_request_with_retry', side_effect=requests.exceptions.RequestException("Network error")):
            with pytest.raises(requests.exceptions.RequestException):
                service.download_pdf("https://arxiv.org/pdf/2301.00006.pdf", "2301.00006")

            # 确保没有留下临时文件
            temp_files = [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]
            assert len(temp_files) == 0

    def test_download_atomic_move(self, service, tmp_path):
        """测试使用原子操作移动文件"""
        mock_response = Mock()
        mock_response.content = create_minimal_valid_pdf()
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = Mock()

        with patch.object(service, '_make_request_with_retry', return_value=mock_response):
            with patch('os.replace') as mock_replace:
                filepath = service.download_pdf("https://arxiv.org/pdf/2301.00007.pdf", "2301.00007")

                # 验证使用了 os.replace 进行原子操作
                mock_replace.assert_called_once()
                args = mock_replace.call_args[0]
                assert args[0].endswith('.tmp')
                assert args[1].endswith('2301.00007.pdf')
