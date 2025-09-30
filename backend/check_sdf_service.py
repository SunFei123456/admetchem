from app.services.sdf_service import SdfService

if __name__ == "__main__":
    with open("input.sdf", "r", encoding="utf-8") as f:
        content = f.read()
    service = SdfService()
    results, total, success, failed = service.sdf_to_smiles(content)
    print({
        "total_molecules": total,
        "successful_conversions": success,
        "failed_conversions": failed,
        "first_smiles": results[0]["smiles"] if results else None,
        "count_results": len(results),
    })
