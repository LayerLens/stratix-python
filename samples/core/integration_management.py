#!/usr/bin/env python3
"""Example: working with integrations via the Stratix SDK.

Requires:
    pip install layerlens
    export LAYERLENS_STRATIX_API_KEY="your-api-key"
"""

from layerlens import Stratix


def main() -> None:
    client = Stratix()

    # --- List all integrations
    response = client.integrations.get_many()

    if response is None or not response.integrations:
        print("No integrations found.")
    else:
        print(f"Found {response.total_count} integration(s):\n")
        for integration in response.integrations:
            print(f"  [{integration.id}] {integration.name}")
            print(f"    Type:    {integration.type}")
            print(f"    Status:  {integration.status}")
            print(f"    Created: {integration.created_at}")
            print()

    # --- List with pagination
    page1 = client.integrations.get_many(page=1, page_size=5)
    if page1:
        print(f"Page 1: showing {page1.count} of {page1.total_count}")

    # --- Get a single integration by ID
    if response and response.integrations:
        integration_id = response.integrations[0].id

        integration = client.integrations.get(integration_id)
        if integration:
            print(f"\nIntegration detail:")
            print(f"  ID:     {integration.id}")
            print(f"  Name:   {integration.name}")
            print(f"  Type:   {integration.type}")
            print(f"  Status: {integration.status}")
            print(f"  Config: {integration.config}")

        # --- Test an integration
        result = client.integrations.test(integration_id)
        if result:
            status = "OK" if result.success else "FAILED"
            print(f"\nTest result: {status}")
            if result.message:
                print(f"  Message: {result.message}")


if __name__ == "__main__":
    main()
