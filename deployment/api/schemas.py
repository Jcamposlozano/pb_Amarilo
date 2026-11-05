"""Contrato REST de la oportunidad y su clasificación de compra."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

Int64 = Annotated[int, Field(strict=True, ge=-(2**63), le=2**63 - 1)]
BinaryLabel = Annotated[int, Field(strict=True, ge=0, le=1)]


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ETAPA_DE_VENTA: str
    FECHA_DE_CREACION_OP: str
    FECHA_ULTIMA_ACTUALIZACION_OP: str
    REFERIDO: str
    TIPO_DE_PROYECTO: str
    METODO_DE_VENTAS: str
    CIUDAD: str
    IMPORTE_TOTAL_OP_PRIMERA_VERSION: FiniteFloat
    DESTINO: str
    TIPO_VENTA: str
    SALA_DE_VENTAS_ALTERNATIVA: str
    ESTADO_PROYECTO: str
    FECHA_INICIO_VENTAS_TRANSACCIONAL: str
    BK_VENDEDOR: Int64
    OBSERVACIONES: str
    OCUPACION_COMPRADOR1: str
    CIUDAD_COMPRADOR1: str

    def to_opportunity(self) -> dict:
        return self.model_dump()


class ClassificationResponse(BaseModel):
    classification: BinaryLabel
