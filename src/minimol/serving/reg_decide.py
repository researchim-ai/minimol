"""Typed Reg Decide API and a small local browser demonstration."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from minimol.decide import RegDecider, RegDecideRequest

DEMO = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Minimol Reg Decide</title>
<style>body{font:16px system-ui;max-width:860px;margin:40px auto;padding:0 20px;background:#fafafa;color:#222}textarea{width:100%;height:110px;font:14px monospace}label{display:block;margin:20px 0 8px}button,select,input{font:inherit;padding:8px}button{cursor:pointer}pre{white-space:pre-wrap;background:#eee;padding:18px}small{color:#555}</style>
<h1>Minimol Reg Decide</h1><p>Одна последовательность — несколько биологических вопросов.</p>
<small>Исследовательский пилот. Поддерживаются K562, HepG2 и SK-N-SH. Вероятности имеют ограниченную development-калибровку.</small>
<label>ДНК (73–200 оснований)</label><textarea id="dna" spellcheck="false" placeholder="Вставьте последовательность ACGTN"></textarea>
<label>Клеточная линия и порог активности (log2FC)</label><select id="assay"><option>K562</option><option>HepG2</option><option>SK-N-SH</option></select> <input id="threshold" type="number" min="-0.5" max="2" step="0.25" value="0.5">
<label>API token, если включена авторизация</label><input id="token" type="password" autocomplete="off">
<p><button id="run">Задать вопросы</button></p><div id="result" aria-live="polite"></div>
<script>document.getElementById('run').onclick=async()=>{const result=document.getElementById('result');result.textContent='Вычисление…';try{const token=document.getElementById('token').value;const response=await fetch('/v1/regulatory/decide',{method:'POST',headers:{'Content-Type':'application/json',...(token?{'Authorization':'Bearer '+token}:{})},body:JSON.stringify({candidates:[{id:'dna',sequence:document.getElementById('dna').value.trim()}],questions:[{id:'above',kind:'noul',candidate_id:'dna',assay:document.getElementById('assay').value,threshold:Number(document.getElementById('threshold').value)},{id:'cell',kind:'assay_choice',candidate_id:'dna',assays:['K562','HepG2','SK-N-SH']},{id:'bins',kind:'score',candidate_id:'dna',assay:document.getElementById('assay').value,bins:[0,0.5,1,2]}]})});const data=await response.json();result.replaceChildren();if(!response.ok||!data.answers){result.textContent=JSON.stringify(data);return}for(const answer of data.answers){const block=document.createElement('pre');const probability=answer.probability===undefined?'':' · '+(100*answer.probability).toFixed(1)+'%';block.textContent=answer.question_id+': '+(answer.abstain?'Отказ: '+answer.reason:String(answer.answer)+probability)+(answer.distribution?'\n'+answer.options.map((name,i)=>name+': '+(answer.distribution[i]*100).toFixed(1)+'%').join('\n'):'');result.append(block)}}catch(error){result.textContent=String(error)}};</script></html>'''


def register_reg_decide(app: FastAPI, decider: RegDecider | None):
    @app.post('/v1/regulatory/decide')
    def predict(body: RegDecideRequest):
        if decider is None:
            return {'abstain': True, 'reason': 'decision_weights_not_loaded', 'research_use_only': True}
        return decider.predict(body)

    @app.get('/reg-decide', response_class=HTMLResponse)
    def demonstration():
        return DEMO
