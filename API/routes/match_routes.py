from fastapi import APIRouter, Depends, HTTPException
from API.schemas.match_schemas import MatchCreate, MatchUpdate, MatchOut
from API.services.match_service import MatchService
from API.utils.response_formatter import success_response
from API.dependencies.auth_dependency import get_current_user

router = APIRouter()

@router.post("/", response_model=dict)
async def create_match(match: MatchCreate, current_user=Depends(get_current_user)):
    new_match = await MatchService.create_match(match)
    return success_response(data=new_match, message="Match created")

@router.get("/", response_model=dict)
async def get_matches():
    matches = await MatchService.get_matches()
    return success_response(data=matches)

@router.get("/{match_id}", response_model=dict)
async def get_match(match_id: int):
    match = await MatchService.get_match(match_id)
    return success_response(data=match)

@router.put("/{match_id}", response_model=dict)
async def update_match(match_id: int, match: MatchUpdate, current_user=Depends(get_current_user)):
    updated = await MatchService.update_match(match_id, match)
    return success_response(data=updated, message="Match updated")

@router.delete("/{match_id}")
async def delete_match(match_id: int, current_user=Depends(get_current_user)):
    await MatchService.delete_match(match_id)
    return success_response(message="Match deleted")
