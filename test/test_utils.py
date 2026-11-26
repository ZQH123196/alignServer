import unittest

from utils import normalize_multipart_line_endings, with_chapter_identity, audio_defect_error_detail


class NormalizeMultipartLineEndingsTests(unittest.TestCase):
    """multipart 字段值的 CRLF 还原。

    Node 的 FormData（undici）把字段值里的 `\\n` 编码成 `\\r\\n`；Python / curl
    客户端直传 `\\n`。服务端入口必须把两种形态都还原成 LF，否则参考文本与调用方
    持有的原文不再逐字相等（2026-10-17 第 4 章：2136 字被当成 2210 字）。
    """

    def test_crlf_is_restored_to_lf(self) -> None:
        # Arrange / Act
        result = normalize_multipart_line_endings("第4章 家族\r\n根据沃尔的研究\r\n")

        # Assert
        self.assertEqual(result, "第4章 家族\n根据沃尔的研究\n")

    def test_lf_input_is_left_alone(self) -> None:
        # Arrange / Act
        result = normalize_multipart_line_endings("甲\n乙\n丙")

        # Assert
        self.assertEqual(result, "甲\n乙\n丙")

    def test_lone_cr_is_treated_as_newline(self) -> None:
        # Arrange: 旧 Mac 行尾
        # Act
        result = normalize_multipart_line_endings("甲\r乙")

        # Assert
        self.assertEqual(result, "甲\n乙")

    def test_empty_text_returns_empty(self) -> None:
        # Act / Assert
        self.assertEqual(normalize_multipart_line_endings(""), "")

    def test_char_count_matches_the_caller_text_after_normalization(self) -> None:
        # Arrange: 事故形态 —— 尾部 chunk 的下标按服务端文本长度算
        text = "第4章 家族\n根据沃尔的研究。\n它的皮毛恢复了完整。"
        received = text.replace("\n", "\r\n")

        # Act
        result = normalize_multipart_line_endings(received)

        # Assert: 长度与调用方原文一致，不再多出换行数个字符
        self.assertEqual(len(result), len(text))
        self.assertEqual(result, text)


class ChapterErrorBodyTests(unittest.TestCase):
    """400 裁决体必须带章节身份与机器可读类别（2026-10-28 第 34 章）。

    现场：覆盖率闸门只回了一段诊断，没有章节身份；调用方拿到的是
    「某次请求中间缺了 116 字」，无法定位是哪一章。且类型与参数错误共用
    `invalid_request_error`，音频缺陷被当成参数问题直接失败。
    """

    def test_identity_is_prepended_to_the_message(self) -> None:
        # Act
        message = with_chapter_identity("音频与参考文本对不上", "merged_某书_zh-Hans_34_第34章.opus")

        # Assert
        self.assertTrue(message.startswith("章节文件：merged_某书_zh-Hans_34_第34章.opus\n"))

    def test_missing_filename_does_not_break_the_prefix(self) -> None:
        # Act
        message = with_chapter_identity("诊断", None)

        # Assert
        self.assertTrue(message.startswith("章节文件：(未命名)\n"))

    def test_audio_defect_body_carries_type_reason_and_file(self) -> None:
        # Act
        detail = audio_defect_error_detail("中间缺了 116 字", "merged_某书_zh-Hans_34_第34章.opus", "coverage-gap")

        # Assert: 类型是调用方唯一的处置开关，不能与 invalid_request_error 混用
        self.assertEqual(detail["error"]["type"], "audio_defect")
        self.assertEqual(detail["error"]["reason"], "coverage-gap")
        self.assertEqual(detail["error"]["file"], "merged_某书_zh-Hans_34_第34章.opus")
        self.assertIn("中间缺了 116 字", detail["error"]["message"])


if __name__ == "__main__":
    unittest.main()