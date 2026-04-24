from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from face_service import (
    register_person,
    recognize_person,
    verify_person,
    list_persons,
    delete_person
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/register")
def register_face(person_id: str = Form(...), image: UploadFile = File(...)):
    try:
        return register_person(person_id, image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recognize")
def recognize_face(image: UploadFile = File(...)):
    try:
        return recognize_person(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/verify/{person_id}")
def verify_face(person_id: str,image: UploadFile = File(...)):
    try:
        return verify_person(person_id, image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/persons")
def get_persons():
    try:
        return list_persons()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/persons/{person_id}")
def remove_person(person_id: str):
    try:
        return delete_person(person_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))