# 의도 라벨&규칙
from enum import Enum
import re

class Intent(str, Enum):
    ACADEMICS = "academics" # 학칙/학사/요람
    NOTICES = "notices" # 공지/식단/일정
    TIPS = "tips" # 대학생활팁
    GENERAL = "general"  # 일반대화
    MICRO_DH = "micro_dh"
    MICRO_IP = "micro_ip"
    MICRO_PL = "micro_pl"

KW = {
  Intent.MICRO_DH: ["디지털휴먼", "digital human"],
  Intent.MICRO_IP: ["지식재산", "IP", "metaverse ip"],
  Intent.MICRO_PL: ["메타버스기획", "metaverse design", "콘텐츠기획"],
}

def rule_intent(text: str) -> Intent | None:
    t = text.lower()
    for intent, kws in KW.items():
        if any(k.lower() in t for k in kws):
            return intent
    if any(k in t for k in ["학칙","규정","수업","수강","졸업","요람","교과목","권장 이수","선수과목"]):
        return Intent.ACADEMICS
    if any(k in t for k in ["식단","메뉴","급식","공지","캘린더","일정","세미나"]):
        return Intent.NOTICES
    if any(k in t for k in ["맛집","약어","팁","공부","동아리"]):
        return Intent.TIPS
    return None