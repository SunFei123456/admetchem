from fastapi import APIRouter
from app.api.endpoints import users, druglikeness, sdf, admet, ai

router = APIRouter()

router.include_router(users.router, prefix="/users", tags=["users"])
router.include_router(
    druglikeness.router, prefix="/druglikeness", tags=["druglikeness"]
)
router.include_router(sdf.router, prefix="/sdf", tags=["sdf"])
router.include_router(admet.router, prefix="/admet", tags=["admet"])
router.include_router(ai.router, prefix="/ai", tags=["ai"])
