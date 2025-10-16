import json
from openai import OpenAI
from config.settings import CONFIG

class FieldExtractor:
    """
    Class dùng để trích xuất các trường thông tin (ID, Name, Birth, Home, Address)
    từ văn bản OCR của giấy tờ tùy thân (CMND/CCCD).
    """

    def __init__(self, api_key=None, model: str = "gpt-4o-mini"):
        """
        Khởi tạo model LLM.
        :param api_key: API key của OpenAI
        :param model: Tên model muốn sử dụng (mặc định: gpt-4o-mini)
        """
        self.client = self.build_client(CONFIG["api_key"])
        self.model = model

    def build_client(self, cfg):
        if cfg["id"]:
            return OpenAI(api_key=cfg["id"])
        else:
            return ValueError(f"No api key input!")

    def extract(self, ocr_text: str) -> dict:
        """
        Gọi model LLM để trích xuất thông tin từ OCR raw text.
        :param ocr_text: Chuỗi text thu được từ OCR
        :return: dict gồm các trường id, name, dob, sex, nationality, placeoforigin, placeofresidence
        """
        prompt = f"""
        Hãy đọc văn bản sau được trích xuất từ căn cước công dân hoặc chứng minh nhân dân Việt Nam,
        và sửa lỗi chính tả nếu có (có thể bao gồm thêm các lỗi về sai tên tỉnh thành, thành phố, đất nước,...), sau đó hãy trích xuất thông tin vào JSON với 7 trường: id, name, dob, sex, nationality, placeoforigin, placeofresidence.

        Nếu không có giá trị cho trường nào thì để "None".
        Trả về JSON thuần, không kèm lời giải thích.

        Văn bản OCR:
        ```
        {ocr_text}
        ```
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        print(response)

        raw_output = response.choices[0].message.content.strip()

        # Nếu model trả JSON trong code block, loại bỏ dấu ```
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`")
            raw_output = raw_output.replace("json", "").strip()

        try:
            data = json.loads(raw_output)
        except Exception:
            # fallback nếu model trả format chưa đúng
            data = {
                "id": "None",
                "name": "None",
                "dob": "None",
                "sex": "None",
                "nationality": "None",
                "placeoforigin": "None",
                "placeofresidence": "None"
            }

        # Đảm bảo đủ khóa
        for key in ["id", "name", "dob", "sex", "nationality", "placeoforigin", "placeofresidence"]:
            if key not in data:
                data[key] = "None"

        return data