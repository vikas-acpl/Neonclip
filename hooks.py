"""Readable opening headings on bright gameplay, blur and facecam layouts."""
import math
KEYS=('hook_duration','hook_background')
def settings(body):
    value=body.get('hook_duration',8);duration=float(8 if value in (None,'') else value)
    if not math.isfinite(duration) or not 2<=duration<=60:raise ValueError('Hook duration must be 2–60 seconds.')
    background=body.get('hook_background',True)
    if not isinstance(background,bool):raise ValueError('Hook background must be on or off.')
    return duration,background

def events(lines,width,height,y,scale,body):
    """ASS background + text, separate layers; no alteration of the source footage."""
    from captions import BS,stamp,safe
    duration,background=settings(body)
    size=round((78 if len(lines)==1 else 72 if len(lines)==2 else 66)*scale)
    pad=max(8,round(26*scale));line_height=round(size*1.22)
    box_w=min(round(width*.94),round(max(len(line) for line in lines)*size*.67)+pad*2)
    box_h=line_height*len(lines)+pad*2
    x=(width-box_w)//2;y=max(pad,min(round(y),height-box_h-pad));end=stamp(duration,True)
    result=[]
    if background:
        shape=f'm 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}'
        tags='{'+BS+f'an7{BS}pos({x},{y}){BS}bord0{BS}shad0{BS}1c&H000000&{BS}1a&H20&{BS}p1'+'}'
        result.append(f'Dialogue: 2,0:00:00.00,{end},Main,,0,0,0,,{tags}{shape}')
    text=(BS+'N').join(safe(line).upper() for line in lines)
    tags='{'+BS+f'an8{BS}pos({width//2},{y+pad}){BS}fs{size}{BS}b1{BS}c&H00FFFFFF&{BS}3c&H00000000&{BS}bord2{BS}shad0{BS}fad(120,180)'+'}'
    result.append(f'Dialogue: 3,0:00:00.00,{end},Main,,0,0,0,,{tags}{text}')
    return result

def attach(app):
    from fastapi import HTTPException
    @app.post('/hooks/suggest')
    def suggest(body:dict):
        import smart,curation
        text=str(body.get('text','')).strip()
        if not 15<=len(text)<=30000:raise HTTPException(400,'Generate or import subtitles for the selected clip first.')
        model=smart._model(True)
        if not model:return {'hook':curation.extract_hook(text),'model':None,'evidence':'Extracted from the supplied transcript'}
        try:return {'hook':smart._heading(model,text),'model':model}
        except Exception as error:raise HTTPException(503,'Local heading model failed. '+str(error)[-160:])
