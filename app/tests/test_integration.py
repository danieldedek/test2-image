import os
import sys
import io
import wave
import struct
import math
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/app')


def _make_wav_bytes(duration_seconds=2.0, sample_rate=16000):
    buf = io.BytesIO()
    n_samples = int(sample_rate * duration_seconds)
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = []
        for i in range(n_samples):
            sample = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.append(struct.pack('<h', sample))
        wf.writeframes(b''.join(frames))
    return buf.getvalue()

class TestGetIndex:

    def test_index_returns_200(self, flask_client):
        response = flask_client.get('/')
        assert response.status_code == 200

    def test_index_contains_asr_demo(self, flask_client):
        response = flask_client.get('/')
        assert b'ASR Demo' in response.data

    def test_index_contains_model_selector(self, flask_client):
        response = flask_client.get('/')
        assert b'canary' in response.data or b'Canary' in response.data

    def test_index_with_model_param(self, flask_client):
        for model in ['whisper', 'canary', 'parakeet']:
            response = flask_client.get(f'/?model={model}')
            assert response.status_code == 200

    def test_index_with_sort_param(self, flask_client):
        for sort in ['name', 'date']:
            response = flask_client.get(f'/?sort={sort}')
            assert response.status_code == 200

    def test_index_with_search_param(self, flask_client):
        response = flask_client.get('/?search=test')
        assert response.status_code == 200

    def test_index_with_page_param(self, flask_client):
        response = flask_client.get('/?page=1')
        assert response.status_code == 200

    def test_index_page_out_of_range(self, flask_client):
        response = flask_client.get('/?page=9999')
        assert response.status_code == 200


class TestFileUpload:

    def test_upload_wav_file(self, flask_client):
        wav_bytes = _make_wav_bytes()
        response = flask_client.post('/', data={
            'file': (io.BytesIO(wav_bytes), 'test_upload.wav'),
            'action': 'upload',
            'selected_model': 'whisper'
        }, content_type='multipart/form-data', follow_redirects=True)
        assert response.status_code == 200

    def test_upload_without_file_does_not_crash(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'upload',
            'selected_model': 'whisper'
        }, content_type='multipart/form-data', follow_redirects=True)
        assert response.status_code == 200

    def test_upload_redirects(self, flask_client):
        wav_bytes = _make_wav_bytes()
        response = flask_client.post('/', data={
            'file': (io.BytesIO(wav_bytes), 'redirect_test.wav'),
            'action': 'upload',
            'selected_model': 'whisper'
        }, content_type='multipart/form-data')
        assert response.status_code in (301, 302)


class TestModelSelection:

    def test_select_whisper(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'select_model',
            'selected_model': 'whisper'
        }, follow_redirects=True)
        assert response.status_code == 200

    def test_select_canary(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'select_model',
            'selected_model': 'canary'
        }, follow_redirects=True)
        assert response.status_code == 200

    def test_select_parakeet(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'select_model',
            'selected_model': 'parakeet'
        }, follow_redirects=True)
        assert response.status_code == 200


class TestClearAction:

    def test_clear_redirects(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'clear',
            'selected_model': 'whisper'
        }, follow_redirects=True)
        assert response.status_code == 200


class TestDownload:

    def test_download_returns_file(self, flask_client):
        response = flask_client.post('/download', data={
            'transcript': 'Hello world test transcript.'
        })
        assert response.status_code == 200
        assert b'Hello world test transcript.' in response.data

    def test_download_empty_transcript(self, flask_client):
        response = flask_client.post('/download', data={
            'transcript': ''
        })
        assert response.status_code == 200

    def test_download_content_disposition(self, flask_client):
        response = flask_client.post('/download', data={
            'transcript': 'Test'
        })
        assert b'attachment' in response.headers.get('Content-Disposition', '').encode()


class TestDeleteFile:

    def test_delete_nonexistent_file_does_not_crash(self, flask_client):
        response = flask_client.post('/delete/nonexistent_file.wav',
                                     follow_redirects=True)
        assert response.status_code == 200

    def test_delete_uploaded_file(self, flask_client, tmp_path):
        """Upload a file, then delete it."""
        wav_bytes = _make_wav_bytes()
        flask_client.post('/', data={
            'file': (io.BytesIO(wav_bytes), 'to_delete.wav'),
            'action': 'upload',
            'selected_model': 'whisper'
        }, content_type='multipart/form-data')

        response = flask_client.post('/delete/to_delete.wav',
                                     follow_redirects=True)
        assert response.status_code == 200


class TestTranscriptionMocked:

    def _mock_transcribe(self, flask_client, engine, extra_data=None):
        mock_asr = MagicMock()
        mock_asr.transcribe.return_value = "mocked transcript result"

        data = {
            'action': 'use_file',
            'selected_model': engine,
            'audio_file': 'fake.wav',
            'device': 'cpu',
            'strategy': 'greedy',
            'beam_size': '5',
        }
        if extra_data:
            data.update(extra_data)

        with patch('main.create_asr_engine', return_value=mock_asr):
            with patch('main.os.path.join', return_value='/fake/fake.wav'):
                with patch('main.secure_filename', return_value='fake.wav'):
                    response = flask_client.post('/', data=data,
                                                 follow_redirects=True)
        return response, mock_asr

    def test_whisper_mocked_transcription(self, flask_client):
        response, mock_asr = self._mock_transcribe(flask_client, 'whisper', {
            'model_size': 'tiny',
            'whisper_language': 'en',
            'temperature': '0.0',
            'best_of': '5',
        })
        assert response.status_code == 200

    def test_canary_mocked_transcription(self, flask_client):
        response, mock_asr = self._mock_transcribe(flask_client, 'canary', {
            'language': 'en',
            'len_pen': '1.0',
        })
        assert response.status_code == 200

    def test_parakeet_mocked_transcription(self, flask_client):
        response, mock_asr = self._mock_transcribe(flask_client, 'parakeet')
        assert response.status_code == 200

    def test_no_audio_file_shows_error(self, flask_client):
        response = flask_client.post('/', data={
            'action': 'use_file',
            'selected_model': 'whisper',
            'audio_file': '',
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'error' in response.data.lower() or b'Error' in response.data


class TestPagination:

    def test_page_1_returns_200(self, flask_client):
        assert flask_client.get('/?page=1').status_code == 200

    def test_page_2_returns_200(self, flask_client):
        assert flask_client.get('/?page=2').status_code == 200

    def test_search_and_sort_combined(self, flask_client):
        assert flask_client.get('/?search=test&sort=name&page=1').status_code == 200

@pytest.mark.slow
@pytest.mark.whisper
class TestWhisperIntegration:

    def test_whisper_transcription_via_flask(self, flask_client, short_wav):
        import shutil, tempfile, os
        upload_dir = flask_client.application.config.get('UPLOAD_FOLDER', '/app/uploads')
        dest = os.path.join(upload_dir, 'integration_test.wav')
        shutil.copy(short_wav, dest)

        response = flask_client.post('/', data={
            'action': 'use_file',
            'selected_model': 'whisper',
            'audio_file': 'integration_test.wav',
            'device': 'cpu',
            'model_size': 'tiny',
            'whisper_language': 'en',
            'strategy': 'greedy',
            'beam_size': '1',
            'temperature': '0.0',
            'best_of': '5',
            'vad_filter': 'on',
            'condition_on_previous_text': 'on',
        }, follow_redirects=True)

        assert response.status_code == 200
        if os.path.exists(dest):
            os.remove(dest)

    def test_whisper_with_timestamps_via_flask(self, flask_client, short_wav):
        import shutil, os
        upload_dir = flask_client.application.config.get('UPLOAD_FOLDER', '/app/uploads')
        dest = os.path.join(upload_dir, 'ts_test.wav')
        shutil.copy(short_wav, dest)

        response = flask_client.post('/', data={
            'action': 'use_file',
            'selected_model': 'whisper',
            'audio_file': 'ts_test.wav',
            'model_size': 'tiny',
            'whisper_language': 'en',
            'beam_size': '1',
            'temperature': '0.0',
            'best_of': '5',
            'show_timestamps': 'on',
        }, follow_redirects=True)

        assert response.status_code == 200
        if os.path.exists(dest):
            os.remove(dest)
