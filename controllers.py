from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse 
from time import ctime

from face_service import (
    register_person,
    recognize_person,
    list_persons,
    delete_person
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok",
            "timestamp": ctime()}


@router.post("/register")
def register_face(person_id: str = Form(...), image: UploadFile = File(...)):
    try:
        result = register_person(person_id, image)
        return JSONResponse(content=result, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recognize")
def recognize_face(image: UploadFile = File(...)):
    try:
        result = recognize_person(image)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

"""
@router.post("/verify/{person_id}")
def verify_face(person_id: str,image: UploadFile = File(...)):
    try:
        result = verify_person(person_id, image)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""

@router.get("/persons")
def get_persons():
    try:
        result = list_persons()
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/persons/{person_id}")
def remove_person(person_id: str):
    try:
        result = delete_person(person_id)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))