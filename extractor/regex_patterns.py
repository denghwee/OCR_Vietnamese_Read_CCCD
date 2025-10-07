import re

# 12 số liên tục cho CCCD
CCCD_REGEX = re.compile(r"\b\d{12}\b")

# Ngày dạng dd/mm/yyyy
DATE_REGEX = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")