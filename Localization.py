import glob
import json
import os

class Localization():
    def __init__(self, root_dir, default_lang):
        self.root = root_dir
        self.locals = self.get_locals(self.root)
        self.current_lang = None
        self.change_language(default_lang)

    def get_locals(self, directory):
        paths = glob.glob(os.path.join(self.root, "*.json"))
        locals = []
        for p in paths:
            lang_code = os.path.basename(p).split(".")[0]
            with open(p, "r") as f:
                json_data = json.load(f)
                name = json_data["metadata"]["name"]
            locals.append({"name": name, 
                            "code": lang_code,
                            "path": p})
        return locals

    def available_locals(self):
        return self.locals

    def current_language(self):
        return self.current_lang["metadata"]["name"]
    
    def current_language_code(self):
        name =  self.current_lang["metadata"]["name"]
        for local in self.locals:
            if local["name"] == name:
                return local["code"]
        return ""

    def current_language_index(self):
        name =  self.current_lang["metadata"]["name"]
        for i, local in enumerate(self.locals):
            if local["name"] == name:
                return i
        return -1

    def change_language(self, lang_code):
        self.current_lang = self.__load_local__(lang_code)

    def __load_local__(self, lang_code):
        found_locals = list(filter(lambda x: x["code"] == lang_code, self.locals))
        
        if len(found_locals) == 0:
            raise ValueError(f"lanage code {lang_code} not exists, there is no such localization file {lang_code}.json")
        
        with open(found_locals[0]["path"], "r") as f:
            return json.load(f)
    
    def localize(self, key):
        if self.current_lang is None:
            return key
        return self.current_lang["data"].get(key, key)
