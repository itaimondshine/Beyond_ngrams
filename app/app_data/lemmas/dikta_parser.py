from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertTokenizerFast,
)
import re
import hebrew_tokenizer as ht


class HebTokenizer:
    def tokenize(self, text):
        tokens = ht.tokenize(text)  # tokenize returns a generator!
        return [token for _, token, _, _ in tokens]


class Parser:
    def parse(self, text):
        raise NotImplementedError()

    def tokenize(self, parse_data):
        raise NotImplementedError()

    def lemmatize(self, parse_data):
        raise NotImplementedError()

    def find_smixuts(self, parse_data):
        raise NotImplementedError()

    def find_coreference_anaphors(self, parse_data):
        raise NotImplementedError()


class DictaParser(Parser):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint")
        self.model = AutoModel.from_pretrained(
            "dicta-il/dictabert-joint", trust_remote_code=True
        )
        self.model.eval()

    def parse(self, text, tokens_limit=0):
        text = text.replace("׳", "")
        if tokens_limit:
            tokens = HebTokenizer().tokenize(text)[:tokens_limit]
            tokens = [
                t
                for t in tokens
                if re.match(r"^[A-Za-z\u0590-\u05FF\uFB1D-\uFB4F0-9\s,.?!%]+$", t)
            ]
            text = " ".join(tokens)
        sentences = re.findall(r".*?[.!?](?:\s|$)|.*$", text)  # text.split(". ")
        # sentences = [f"{s}." for s in sentences]

        ud_results = self.model.predict(sentences, self.tokenizer, output_style="ud")
        json_datas = self.model.predict(sentences, self.tokenizer, output_style="json")

        ud_trees = []
        for i, ud_result in enumerate(ud_results):
            headers = [
                "ID",
                "FORM",
                "LEMMA",
                "UPOS",
                "XPOS",
                "FEATS",
                "HEAD",
                "DEPREL",
                "DEPS",
                "MISC",
            ]
            ud_tree = [dict(zip(headers, item.split("\t"))) for item in ud_result[2:]]
            ud_trees.append(ud_tree)
            for x in ud_tree:
                x["sentence"] = i

        for i, json_data in enumerate(json_datas):
            for x in json_data["tokens"]:
                x["sentence"] = str(i)

        result = dict(ud_trees=ud_trees, json_datas=json_datas)
        return result

    def tokenize(self, parse_data):
        json_datas = parse_data["json_datas"]
        result = []
        for json_data in json_datas:
            tokens = [x["token"] for x in json_data["tokens"]]
            result += tokens
        return result

    def lemmatize(self, parse_data, morphemes=False):
        ud_trees = parse_data["ud_trees"]
        result = []
        for ud_tree in ud_trees:
            key = "LEMMA"
            lemmas = [x[key] for x in ud_tree if x[key] != "_"]
            result += lemmas
        return result

    def find_smixuts(self, parse_data):
        ud_trees = parse_data["ud_trees"]
        smixuts = set()
        for ud_tree in ud_trees:
            for node in ud_tree:
                if node["DEPREL"] == "compound:smixut":
                    head = self._get_head_node(
                        ud_tree, node, ignore_deprel="compound:smixut"
                    )
                    smixuts.add(head["ID"] if head else "root")
        return list(smixuts)

    def find_coreference_anaphors(self, parse_data):
        ud_trees = parse_data["ud_trees"]
        result = []
        for ud_tree in ud_trees:
            coref_anaphors = [n for n in ud_tree if n["UPOS"] in ["PRON"]]
            result += coref_anaphors
        return result

    def _get_head_node(self, ud_tree, node, ignore_deprel=None):
        head_idx = node["HEAD"]
        head = next((n for n in ud_tree if n["ID"] == head_idx), None)
        if head is None:
            return None
        elif head["DEPREL"] != ignore_deprel:
            return head
        else:
            return self._get_head_node(
                ud_tree=ud_tree, node=head, ignore_deprel=ignore_deprel
            )
