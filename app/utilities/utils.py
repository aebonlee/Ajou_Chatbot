import re
def de_newline(s:str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\r"," ")).strip()