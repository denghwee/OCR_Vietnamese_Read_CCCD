import re
from .regex_patterns import CCCD_REGEX, DATE_REGEX

class FieldExtractor:
    def __init__(self):
        pass

    def extract(self, ocr_lines):
        data = {
            "id_number": None,
            "name": None,
            "dob": None,
            "gender": None,
            "nationality": None,
            "hometown": None,
            "residence": None,
            "expiry": None,
        }

        n = len(ocr_lines)
        for i, line in enumerate(ocr_lines):
            clean = line.strip()

            # --- 1. Số CCCD ---
            if data["id_number"] is None:
                m = CCCD_REGEX.search(clean.replace(" ", ""))
                if m:
                    data["id_number"] = m.group()

            # --- 2. Họ tên ---
            if any(k in clean for k in ["Họ và tên", "Full name"]):
                if i + 1 < n:
                    data["name"] = ocr_lines[i + 1].strip()

            # --- 3. Ngày sinh ---
            if any(k in clean for k in ["Ngày", "Ngày, tháng, năm sinh", "Ngày sinh", "Date of birth", "Date of binth"]):
                m = DATE_REGEX.search(clean)
                if not m:
                    j = 0
                    while j + 1 < n:
                        m = DATE_REGEX.search(ocr_lines[j + 1])
                        j += 1 
                        if m:
                            break
                if m:
                    data["dob"] = m.group()

            # --- 4. Quốc tịch ---
            if "Quốc tịch" in clean or "Nationality" in clean:
                if "Việt Nam" in clean:
                    data["nationality"] = "Việt Nam"
                elif i + 1 < n:
                    data["nationality"] = ocr_lines[i + 1].strip()

            # --- 5. Giới tính ---
            if "Giới tính" in clean or "Sex" in clean:
                if "Nam" in clean or "nam" in clean:
                    data["gender"] = "Nam"
                elif "Nữ" in clean or "nữ" in clean:
                    data["gender"] = "Nữ"
                else:
                    j = 0
                    while j + 1 < n:
                        g = ocr_lines[j + 1].strip()
                        if g in ["Nam", "Nữ", "nam", "nữ"]:
                            data["gender"] = g
                            break
                        j += 1

            # --- 6. Quê quán ---
            if "Quê quán" in clean or "Place of origin" in clean:
                if i + 1 < n:
                    data["hometown"] = ocr_lines[i + 1].strip()

            # --- 7. Nơi thường trú ---
            if "Nơi thường trú" in clean or "Place of residence" in clean:
                if i + 1 < n:
                    addr = ocr_lines[i + 1].strip()
                    if i + 2 < n and not any(k in ocr_lines[i + 2] for k in ["giá trị", "hết hạn"]):
                        addr += " " + ocr_lines[i + 2].strip()
                    data["residence"] = addr

            # --- 8. Hạn sử dụng ---
            if "giá trị" in clean or "hết hạn" in clean:
                m = DATE_REGEX.search(clean)
                if not m and i + 1 < n:
                    m = DATE_REGEX.search(ocr_lines[i + 1])
                if m:
                    data["expiry"] = m.group()
                else:
                    # fallback: tìm năm trong câu
                    year_match = re.search(r"\b20\d{2}\b", clean)
                    if year_match:
                        data["expiry"] = "??/??/" + year_match.group()

        return data
