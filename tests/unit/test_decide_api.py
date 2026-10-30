import asyncio

import httpx

from minimol.serving.app import create_app


def test_missing_weights_and_auth_on_decision_route():
    async def run():
        app = create_app(token='token')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            payload = {'candidates': [{'id': 'x', 'sequence': 'ACGT'}],
                       'questions': [{'id': 'q', 'kind': 'noul', 'candidate_id': 'x', 'assay': 'K562', 'threshold': 0}]}
            assert (await client.post('/v1/regulatory/decide', json=payload)).status_code == 401
            response = await client.post('/v1/regulatory/decide', json=payload, headers={'Authorization': 'Bearer token'})
            assert response.json()['reason'] == 'decision_weights_not_loaded'
            assert (await client.get('/reg-decide', headers={'Authorization': 'Bearer token'})).status_code == 200
    asyncio.run(run())
