"""Module management endpoints."""

from fastapi import APIRouter
from api.schemas import ModuleInfo

router = APIRouter(prefix="/modules", tags=["modules"])

AVAILABLE_MODULES = [
    ModuleInfo(name="epp_gloves", status="active", description="EPP Guantes — con_guante / sin_guante"),
    ModuleInfo(name="inactivity", status="active", description="Inactividad — estado_activo / estado_inactivo / estado_preparacion / estado_descarga"),
    ModuleInfo(name="mixed", status="active", description="Mixed EPP + operario — mano_con_guante / mano_sin_guante / operario_activo / operario_transicion"),
    ModuleInfo(name="metrology", status="active", description="Metrología — Inspección dimensional bracket OK / NOK"),
    ModuleInfo(name="kitting", status="active", description="Kitting — Conteo de kits etiquetados (operario / bolsa_kit / kit_listo / hoja_etiquetas)"),
    ModuleInfo(name="procesado", status="active", description="Procesado de piezas — Conteo de piezas procesadas (pieza_ok / pieza_sin_proceso)"),
]


@router.get("", response_model=list[ModuleInfo], summary="List all modules")
async def list_modules():
    return AVAILABLE_MODULES
