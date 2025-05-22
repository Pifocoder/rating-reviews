from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def filter_nouns(text: str, keywords: list) -> list:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    # Create set of noun lemmas
    nouns = set()
    for token in doc.tokens:
        if token.pos == 'NOUN':
            token.lemmatize(morph_vocab)
            nouns.add(token.lemma.lower())
    
    # Filter keywords that are in nouns set
    return [kw.lower() for kw in keywords if kw.lower() in nouns]
