from io import BytesIO
import base64
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4
import wave

from fastapi.testclient import TestClient
from PIL import Image
import httpx
import pytest

from opendirector.application.prompt_optimization import SKILLS
from opendirector.infrastructure.local_projects import LocalProjects
from opendirector.infrastructure.model_configs import ModelConfigs
from opendirector.integrations.chat_completions import ChatCompletions
from opendirector.installation import builtin_skills
from opendirector_api.main import create_app


@pytest.mark.merge
def test_rewrite_uses_actual_project_assets_and_h3_skill_without_generating(tmp_path, monkeypatch):
    received = []
    outputs = iter(['Keep the shot [[OD_REF_1]] and the spoken line.', 'Lost the inline reference.', 'Improved image.'])

    def model(request):
        received.append(json.loads(request.content))
        answer = next(outputs)
        delta = {'choices': [{'delta': {'content': answer}, 'finish_reason': 'stop'}]}
        return httpx.Response(200, text=f'data: {json.dumps(delta)}\n\ndata: [DONE]\n\n')

    local = LocalProjects(tmp_path)
    configs = ModelConfigs(tmp_path)
    text = configs.save('agent', 'http://127.0.0.1:12345/v1', 'text-only', '', auth_mode='none')
    vision = configs.save('agent', 'http://127.0.0.1:12345/v1', 'vision-text', '', auth_mode='none',
                          input_modalities=['text', 'image'])
    video = configs.save('video', 'http://127.0.0.1:12345/v1', 'FastH3-Omni-INT8', '',
                         protocol='vllm_omni', connection_type='self_hosted', auth_mode='none',
                         video_adapter='fasth3')
    project, node_id, spec_id, reference_id, image_id, upload = map(str, (uuid4() for _ in range(6)))
    root = f'/api/creation/projects/{project}'
    app = create_app(local, chat_provider=ChatCompletions(httpx.MockTransport(model)))
    with TestClient(app, headers={'X-OpenDirector-Request': '1', 'X-OpenDirector-Session': str(uuid4())}) as c:
        def command(kind, payload):
            response = c.post(root + '/commands', json=dict(type=kind, projectId=project, commandId=str(uuid4()), payload=payload))
            assert response.status_code == 200 and response.json()['status'] == 'applied', response.text
            return response.json()

        command('project.create', {'name': 'Prompt optimization'})
        assert c.post(root + '/open').status_code == 200
        with BytesIO() as buf:
            Image.new('RGB', (64, 64), 'purple').save(buf, format='PNG')
            assert c.put(root + '/uploads/' + upload, content=buf.getvalue(),
                         headers={'X-Asset-Name': 'reference.png'}).status_code == 200
        command('asset.import', {'uploadHandle': upload, 'name': 'reference.png', 'expectedKind': 'image'})
        command('asset.writeText', {'name': 'notes.md', 'text': 'Keep the exact spoken line.'})
        before = c.get(root).json()
        assets = {a['kind']: {**{k: a[k] for k in ('kind', 'revision', 'fingerprint')}, 'assetId': a['id']}
                  for a in before['assets']}
        operations = [
            {'type': 'generationSpec.create', 'payload': {'spec': {'id': spec_id, 'projectId': project,
             'kind': 'video', 'modelConfigId': video['id'], 'parameters': {'videoMode': 'reference', 'durationSeconds': 5}}}},
            {'type': 'canvas.show', 'payload': {'item': {'id': node_id, 'projectId': project, 'kind': 'video',
             'position': {'x': 100, 'y': 50}, 'binding': {'role': 'generation',
             'objectRef': {'objectType': 'generation_spec', 'id': spec_id}}}}},
            {'type': 'canvas.show', 'payload': {'item': {'id': image_id, 'projectId': project, 'kind': 'image',
             'position': {'x': 0, 'y': 50}, 'binding': {'role': 'empty'}}}},
            *({'type': 'reference.add', 'payload': {'reference': {'id': rid, 'projectId': project,
                 'targetSpecId': spec_id, 'binding': 'pinned', 'source': assets[kind], 'order': order}}}
              for order, (kind, rid) in enumerate((('image', reference_id), ('text', str(uuid4()))))),
        ]
        command('creation.batch', {'operations': operations})
        before = c.get(root).json()
        response = c.post(root + f'/items/{node_id}/optimize-prompt',
                          json={'prompt': 'Continue with [[OD_REF_1]]', 'inlineReferences': [reference_id]})
        assert response.status_code == 200, response.text
        assert response.json() == {'prompt': 'Keep the shot [[OD_REF_1]] and the spoken line.',
                                   'skill': SKILLS['fasth3-omni'], 'audioOmitted': False}
        first = received[0]
        assert first['model'] == vision['modelId']
        rules = (builtin_skills() / SKILLS['fasth3-omni'] / 'SKILL.md').read_text(encoding='utf-8')
        assert rules in first['messages'][0]['content']
        parts = first['messages'][1]['content']
        assert 'Continue with [[OD_REF_1]]' in parts[0]['text']
        assert any(p.get('text') == 'Keep the exact spoken line.' for p in parts)
        data_url = next(p['image_url']['url'] for p in parts if p['type'] == 'image_url')
        with Image.open(BytesIO(base64.b64decode(data_url.split(',')[1]))) as input_image:
            assert input_image.size == (64, 64)
        response = c.post(root + f'/items/{node_id}/optimize-prompt',
                          json={'prompt': 'Continue with [[OD_REF_1]]', 'inlineReferences': [reference_id]})
        assert response.status_code == 502 and response.json()['code'] == 'prompt_optimization_result'
        response = c.post(root + f'/items/{image_id}/optimize-prompt', json={'prompt': 'An image at dusk.'})
        assert response.status_code == 200 and response.json()['skill'] is None
        assert received[-1]['model'] == text['modelId']
        rewrite_url = root + f'/items/{node_id}/optimize-prompt'
        draft = {'prompt': 'Continue with [[OD_REF_1]]', 'inlineReferences': [reference_id]}
        response = c.post(rewrite_url, json={**draft, 'assistantModel': {'runtime': 'default', 'modelId': text['id']}})
        assert response.status_code == 409 and response.json()['code'] == 'prompt_optimization_vision'
        response = c.post(rewrite_url, json={**draft, 'assistantModel': {'runtime': 'deepseek'}})
        assert response.status_code == 409 and response.json()['code'] == 'prompt_optimization_cli_unsupported'

        async def rewrite(runtime, model_id, thinking, messages):
            assert (runtime, model_id, thinking) == ('codex', None, 'high')
            assert rules in messages[0]['content']
            assert any(part['type'] == 'image_url' for part in messages[1]['content'])
            return 'Native model prompt with [[OD_REF_1]].'

        monkeypatch.setattr('opendirector.agent.prompt_rewrite.rewrite', rewrite)
        response = c.post(rewrite_url, json={**draft,
            'assistantModel': {'runtime': 'codex', 'runtimeModel': None, 'thinking': 'high'}})
        assert response.status_code == 200, response.text
        assert response.json()['prompt'] == 'Native model prompt with [[OD_REF_1]].'
        assert len(received) == 3
        assert c.get(root).json() == before


@pytest.mark.merge
def test_seedance_video_draft_uses_the_seedance_skill(tmp_path):
    received = []

    def model(request):
        received.append(json.loads(request.content))
        delta = {'choices': [{'delta': {'content': 'A five second product shot.'}, 'finish_reason': 'stop'}]}
        return httpx.Response(200, text=f'data: {json.dumps(delta)}\n\ndata: [DONE]\n\n')

    local = LocalProjects(tmp_path)
    configs = ModelConfigs(tmp_path)
    configs.save('agent', 'http://127.0.0.1:12345/v1', 'text-only', '', auth_mode='none')
    video = configs.save('video', 'https://ark.cn-beijing.volces.com/api/plan/v3',
                         'doubao-seedance-2.0', 'key')
    project, node_id, spec_id = map(str, (uuid4() for _ in range(3)))
    root = f'/api/creation/projects/{project}'
    app = create_app(local, chat_provider=ChatCompletions(httpx.MockTransport(model)))
    with TestClient(app, headers={'X-OpenDirector-Request': '1', 'X-OpenDirector-Session': str(uuid4())}) as c:
        def command(kind, payload):
            response = c.post(root + '/commands', json=dict(type=kind, projectId=project,
                              commandId=str(uuid4()), payload=payload))
            assert response.status_code == 200 and response.json()['status'] == 'applied', response.text

        command('project.create', {'name': 'Seedance prompt optimization'})
        assert c.post(root + '/open').status_code == 200
        command('creation.batch', {'operations': [
            {'type': 'generationSpec.create', 'payload': {'spec': {'id': spec_id, 'projectId': project,
             'kind': 'video', 'modelConfigId': video['id'], 'purpose': 'generation',
             'parameters': {'videoMode': 'reference', 'durationSeconds': 5,
                            'aspectRatio': '16:9', 'resolution': '720P'}}}},
            {'type': 'canvas.show', 'payload': {'item': {'id': node_id, 'projectId': project, 'kind': 'video',
             'position': {'x': 0, 'y': 0}, 'binding': {'role': 'generation',
             'objectRef': {'objectType': 'generation_spec', 'id': spec_id}}}}},
        ]})
        response = c.post(root + f'/items/{node_id}/optimize-prompt', json={'prompt': '一只猫在公园里玩球'})
        assert response.status_code == 200, response.text
        assert response.json()['skill'] == 'seedance-prompt-optimization'
        system = received[0]['messages'][0]['content']
        seedance = (builtin_skills() / 'seedance-prompt-optimization' / 'SKILL.md').read_text(encoding='utf-8')
        assert seedance in system
        h3 = (builtin_skills() / SKILLS['vllm-omni-h3'] / 'SKILL.md').read_text(encoding='utf-8')
        assert h3 not in system


@pytest.mark.merge
@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='Requires FFmpeg')
def test_video_and_audio_delivery_follows_selected_chat_model_inputs(tmp_path):
    received = []

    def model(request):
        received.append(json.loads(request.content))
        delta = {'choices': [{'delta': {'content': 'Keep this framing.'}, 'finish_reason': 'stop'}]}
        return httpx.Response(200, text=f'data: {json.dumps(delta)}\n\ndata: [DONE]\n\n')

    wav = tmp_path / 'audio-reference.wav'
    with wave.open(str(wav), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b'\x00\x00' * 16000)
    fixture = Path(__file__).parent / 'fixtures' / 'timeline-text' / 'visual-a.mp4'
    clip = tmp_path / 'video-reference.mp4'
    clip.write_bytes(subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(fixture), '-i', str(wav), '-map', '0:v:0',
        '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
        '-movflags', '+frag_keyframe+empty_moov+default_base_moof', '-f', 'mp4', 'pipe:1',
    ]))

    local = LocalProjects(tmp_path)
    configs = ModelConfigs(tmp_path)
    models = {name: configs.save('agent', 'http://127.0.0.1:12345/v1', name, '', auth_mode='none',
                                 input_modalities=['text', *inputs])
              for name, inputs in [('frames-only', ['image']), ('video-only', ['video']),
                                   ('full-media', ['video', 'audio'])]}
    project, node, spec = map(str, (uuid4() for _ in range(3)))
    endpoint = f'/api/creation/projects/{project}'
    app = create_app(local, chat_provider=ChatCompletions(httpx.MockTransport(model)))
    with TestClient(app, headers={'X-OpenDirector-Request': '1', 'X-OpenDirector-Session': str(uuid4())}) as c:
        def command(kind, payload):
            response = c.post(endpoint + '/commands', json={'type': kind, 'projectId': project,
                              'commandId': str(uuid4()), 'payload': payload})
            assert response.status_code == 200 and response.json()['status'] == 'applied', response.text

        command('project.create', {'name': 'Media input'})
        assert c.post(endpoint + '/open').status_code == 200
        for file, kind in [(clip, 'video'), (wav, 'audio')]:
            upload = str(uuid4())
            response = c.put(endpoint + '/uploads/' + upload, content=file.read_bytes(),
                             headers={'X-Asset-Name': file.name})
            assert response.status_code == 200, response.text
            command('asset.import', {'uploadHandle': upload, 'name': file.name, 'expectedKind': kind})
        assets = c.get(endpoint).json()['assets']
        command('creation.batch', {'operations': [
            {'type': 'generationSpec.create', 'payload': {'spec': {'id': spec, 'projectId': project,
             'kind': 'video', 'parameters': {}}}},
            {'type': 'canvas.show', 'payload': {'item': {'id': node, 'projectId': project, 'kind': 'video',
             'position': {'x': 0, 'y': 0}, 'binding': {'role': 'generation',
             'objectRef': {'objectType': 'generation_spec', 'id': spec}}}}},
            *({'type': 'reference.add', 'payload': {'reference': {'id': str(uuid4()), 'projectId': project,
               'targetSpecId': spec, 'binding': 'pinned', 'order': order,
               'source': {'assetId': asset['id'], **{k: asset[k] for k in ('kind', 'revision', 'fingerprint')}}}}}
              for order, asset in enumerate(assets)),
        ]})

        def optimize(preferred=None, *, audio_omitted):
            draft = {'prompt': 'Continue the shot.'}
            if preferred:
                draft['preferredModelId'] = models[preferred]['id']
            response = c.post(endpoint + f'/items/{node}/optimize-prompt', json=draft)
            assert response.status_code == 200, response.text
            assert response.json()['audioOmitted'] is audio_omitted
            return received[-1]['messages'][1]['content']

        parts = optimize(audio_omitted=False)
        assert received[-1]['model'] == 'full-media'
        assert sorted(p['type'] for p in parts if p['type'] != 'text') == ['input_audio', 'video_url']
        video_url = next(p['video_url']['url'] for p in parts if p['type'] == 'video_url')
        assert base64.b64decode(video_url.split(',', 1)[1]) == clip.read_bytes()
        audio = next(p['input_audio'] for p in parts if p['type'] == 'input_audio')
        assert audio['format'] == 'wav' and base64.b64decode(audio['data']) == wav.read_bytes()

        parts = optimize('video-only', audio_omitted=True)
        assert [p['type'] for p in parts if p['type'] != 'text'] == ['video_url']
        assert wav.name not in json.dumps(parts)
        muted = tmp_path / 'muted.mp4'
        muted.write_bytes(base64.b64decode(next(p['video_url']['url'] for p in parts if p['type'] == 'video_url').split(',', 1)[1]))

        def streams(file):
            value = subprocess.check_output(['ffprobe', '-v', 'error', '-of', 'json', '-count_frames',
                                             '-show_entries', 'stream=codec_type,duration,nb_read_frames', str(file)])
            return json.loads(value)['streams']

        assert [value['codec_type'] for value in streams(clip)] == ['video', 'audio']
        assert streams(muted) == streams(clip)[:1]

        parts = optimize('frames-only', audio_omitted=True)
        assert [p['type'] for p in parts if p['type'] != 'text'] == ['image_url'] * 3
        assert wav.name not in json.dumps(parts)
