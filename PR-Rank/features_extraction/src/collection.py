import json
from collections import defaultdict
from publicsuffix2 import fetch, get_sld


class Collection(object):
    def __init__(self):
        """
        Compute the following statistics
        df: document frequency
        cf: collection frequency
        dn: total number of documents
        cn: total number of words
        """
        self.df = defaultdict(int)
        self.cf = defaultdict(int)
        self.dn = 0
        self.cn = 0

    def add(self, token_freq):
        for w in token_freq.keys():
            self.cf[w] += token_freq[w]
            self.cn += token_freq[w]
            self.df[w] += 1
        self.dn += 1

    def dump(self, f):
        json.dump(self.to_json(), f)

    def to_json(self):
        return {"df": self.df, "cf": self.cf, "dn": self.dn, "cn": self.cn}

    @classmethod
    def load(cls, f):
        data = json.load(f)
        collection = Collection()
        for key in data:
            setattr(collection, key, data[key])
        return collection

    @property
    def avgdlen(self):
        return float(self.cn) / self.dn


class DomainCounter:
    def __init__(self):
        self.domain_freq = defaultdict(int)
        self.psl_file = fetch()

    def add(self, url: str):
        # url に必ず http:// がついている？
        domain = url.split("/")[2]
        sld = get_sld(domain, self.psl_file)
        self.domain_freq[sld] += 1

    def dump(self, f):
        json.dump(self.to_json(), f)

    def to_json(self):
        return self.domain_freq

    @classmethod
    def load(cls, f):
        domain_freq = json.load(f)
        domain_counter = DomainCounter()
        setattr(domain_counter, "domain_freq", domain_freq)
        return domain_counter
