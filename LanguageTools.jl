module LanguageTools

export tokenize

diacritics = "\u0300-\u036F"
tok_re = Regex("[a-z]+[:.].*?(?=\\s)|[\\w]+|[\\w-]+[\\w]+|[^\\w\\s]")
accent_re = Regex("[$(LanguageTools.diacritics)]")

function load(filepath)
    lemma_dict_file = open(filepath, "r")
    lemma_dict = Dict()
    for line in eachline(lemma_dict_file)
        parts = split(strip(line), "\t")
        lemma_dict[parts[1]] = parts[2]
    end
    return lemma_dict
end

function lemmatize(dict, word)
    get(dict, word, word)
end


function tokenize(line)
    # cyrillic = "\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u1D2B\u1D78\uFE2E\uFE2F"
    # re = Regex("[\\w$diacritics][\\w$diacritics-]+[\\w$diacritics]|[\\w$diacritics]+|[^\\w\\s]")
    
    matches = eachmatch(tok_re, line)
    tokens = [m.match for m in matches]
    return tokens
end

function remove_accents(line)
    return replace(line, accent_re  => "")
end

end

# en_lemm = Lemmatizer.load("en_lemma.dict")
# println(en_lemm["it"])
# println(en_lemm["has"])

# println(Lemmatizer.tokenize("a bb cbb"))
# println(LanguageTools.tokenize("Манхэ́ттен[1] (англ. Manhattan [mænˈhætən]) — историческое ядро города Нью-Йорка и одно из его пяти боро. Кроме острова Манхэттен, боро включает в себя несколько небольших островов (см. География Манхэттена). "))
