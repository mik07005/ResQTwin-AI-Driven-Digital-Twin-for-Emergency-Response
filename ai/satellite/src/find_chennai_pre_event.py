import requests

CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
)

products = [
    "S1A_IW_GRDH_1SDV_20211104T003212_20211104T003237_040414_04CA78_20F6.SAFE",
    "S1A_IW_GRDH_1SDV_20211104T003147_20211104T003212_040414_04CA78_AD56.SAFE",
]

for name in products:
    print("\n" + "=" * 80)
    print("Searching:")
    print(name)

    params = {
        "$filter": f"Name eq '{name}'",
        "$top": 10,
    }

    try:
        response = requests.get(
            CATALOGUE_URL,
            params=params,
            timeout=60,
        )

        print("HTTP status:", response.status_code)

        response.raise_for_status()

        data = response.json()
        results = data.get("value", [])

        if not results:
            print("NO PRODUCT FOUND")
            continue

        for product in results:
            print("\nPRODUCT FOUND")
            print("ID:       ", product.get("Id"))
            print("Name:     ", product.get("Name"))
            print("S3 Path:  ", product.get("S3Path"))
            print("Size:     ", product.get("ContentLength"))
            print("Start:    ", product.get("ContentDate", {}).get("Start"))

    except Exception as e:
        print("ERROR:", repr(e))

print("\n" + "=" * 80)
print("SEARCH COMPLETE")