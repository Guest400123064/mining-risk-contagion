from typing import List, Dict, NoReturn, Any, Callable


from src.text import text_to_sentences, normalize_text


class ComponentTextViewer:

    def __init__(self, 
                 text: str,
                 topic_predict_fn: Callable[[str], List[int]],
                 noise_predict_fn: Callable[[str], int],
                 action_predict_fn: Callable[[str], int]) -> NoReturn:

        self.component_text = normalize_text(text)
        self.sentences_info = []

        for sentence in text_to_sentences(self.component_text):
            self.sentences_info.append({
                "text":       sentence,
                "topic_tags": topic_predict_fn(sentence),
                "is_noise":   noise_predict_fn(sentence)
            })

    def get_num_total_sentences(self) -> int:
        return len(self.sentences_info)
    
    def get_num_noise_sentences(self) -> int:
        ret = 0
        for info in self.sentences_info:
            ret += info["is_noise"]
        return ret

    def get_num_topic_sentences(self, topic_id: int) -> int:
        ret = 0
        for info in self.sentences_info:
            ret += int(topic_id in info["topic_tags"])
        return ret
