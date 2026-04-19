# Benchmark 360 — Roadmap de Despliegue en AWS

## Vision: De Hackathon a Produccion Enterprise

Este documento describe la ruta de migracion del pipeline containerizado
a una infraestructura AWS completamente gestionada.

---

## Arquitectura Propuesta en AWS

```
                    EventBridge (Cron 06:00 AM)
                            |
                            v
                    +----------------+
                    |  AWS Lambda    |
                    |  (Trigger)     |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    |  AWS ECS       |
                    |  Fargate       |
                    |  (Container)   |
                    +-------+--------+
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
        +---------+   +---------+   +---------+
        |   S3    |   |   S3    |   |   SNS   |
        | Parquet |   |  Cache  |   | Alertas |
        +---------+   +---------+   +----+----+
              |                          |
              v                     +----+----+
        +-----------+               |  Email  |
        |  Athena   |               |  Slack  |
        |  (Query)  |               +---------+
        +-----+-----+
              |
              v
        +-----------+
        | QuickSight|
        | Dashboard |
        +-----------+
```

## Servicios AWS Recomendados

| Componente | Servicio AWS | Justificacion |
|---|---|---|
| Container Runtime | **ECS Fargate** | Serverless, sin gestionar EC2. Paga solo por ejecucion. |
| Almacenamiento Parquet | **S3** | Costo ~$0.023/GB/mes. Parquet + Snappy es nativo de S3. |
| Programacion Diaria | **EventBridge** | Cron nativo de AWS, sin servidor. |
| Cache LLM | **S3 + DynamoDB** | DynamoDB para lookup rapido, S3 para respuestas completas. |
| Consultas SQL | **Athena** | Queries sobre Parquet en S3 sin infraestructura. |
| Dashboard Ejecutivo | **QuickSight** | BI nativo de AWS, conecta directo a Athena. |
| Alertas | **SNS + Lambda** | Notificaciones a Slack/Email cuando se detectan cambios. |
| Secrets | **Secrets Manager** | API keys de LLMs almacenadas de forma segura. |
| Logs | **CloudWatch** | Logs centralizados con alertas automaticas. |
| CI/CD | **GitHub Actions + ECR** | Build imagen → Push a ECR → Deploy a ECS. |

## Estimacion de Costos Mensuales

| Servicio | Uso Estimado | Costo/Mes |
|---|---|---|
| ECS Fargate | 1 tarea/dia x 10 min x 0.5 vCPU | ~$1.50 |
| S3 | 100 MB datos + cache | ~$0.10 |
| EventBridge | 30 invocaciones/mes | ~$0.00 |
| Athena | 10 queries/dia x 10 MB | ~$0.50 |
| SNS | 30 notificaciones/mes | ~$0.00 |
| Secrets Manager | 5 secrets | ~$2.00 |
| **TOTAL** | | **~$4.10/mes** |

> **Comparacion:** Un analista dedicado a este trabajo cuesta ~$800/mes.
> Benchmark 360 en AWS cuesta $4.10/mes. ROI: **195x**.

## Fases de Migracion

### Fase 1: Containerizacion (COMPLETADA)
- Dockerfile multi-stage optimizado
- docker-compose para ejecucion local
- .dockerignore para builds eficientes

### Fase 2: Push a ECR
```bash
# Crear repositorio en ECR
aws ecr create-repository --repository-name benchmark-360

# Build y push
docker build -t benchmark-360 .
docker tag benchmark-360:latest [ACCOUNT].dkr.ecr.[REGION].amazonaws.com/benchmark-360:latest
docker push [ACCOUNT].dkr.ecr.[REGION].amazonaws.com/benchmark-360:latest
```

### Fase 3: ECS Fargate Task
- Crear Task Definition con la imagen de ECR
- Configurar variables de entorno desde Secrets Manager
- Asignar 0.5 vCPU + 1GB RAM (suficiente para el pipeline)

### Fase 4: EventBridge Schedule
```json
{
  "ScheduleExpression": "cron(0 11 * * ? *)",
  "Description": "Benchmark 360 - 06:00 AM Ecuador"
}
```

### Fase 5: S3 + Athena + QuickSight
- Output del Parquet directo a S3
- Crear tabla Athena sobre la particion anio/mes/dia
- Conectar QuickSight para dashboards ejecutivos

---

## Seguridad en AWS

| Control | Implementacion |
|---|---|
| API Keys | AWS Secrets Manager (nunca en variables de entorno) |
| Network | VPC privada, sin acceso publico |
| IAM | Roles con minimo privilegio |
| Encryption | S3 SSE-S3, DynamoDB encryption at rest |
| Audit | CloudTrail para todas las operaciones |
