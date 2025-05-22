import yake

def extract_keywords(text: str, lang: str = "ru", top: int = 40) -> list:
    kw_extractor = yake.KeywordExtractor(
        lan=lang,
        n=1,
        top=top,
        dedupLim=0.3,
        dedupFunc='seqm'
    )
    keywords = [kw[0] for kw in kw_extractor.extract_keywords(text)]
    return keywords
