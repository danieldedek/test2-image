from abc import ABC, abstractmethod

class BaseASR(ABC):
    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass
