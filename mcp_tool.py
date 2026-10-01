"""
mcp_tool.py
===========
Lightweight MCP (Model Context Protocol) Server & Tool Integration.

Usage:
  1. Standalone MCP Server:
       python mcp_tool.py --transport stdio
       python mcp_tool.py --transport sse --port 8090

  2. Integrate into your EXISTING MCP Project:
       from aadhaar_validator import AadhaarValidator
       validator = AadhaarValidator(model_path="models/gliner_model")

       @mcp.tool()
       def validate_aadhaar(base64_data: str) -> dict:
           return validator.process_base64(base64_data)
"""

import sys
import argparse
from typing import Dict, Any, Optional

try:
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("aadhaar-mcp-validator")
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("aadhaar-mcp-validator")

from aadhaar_validator import AadhaarValidator, get_validator


@mcp.tool()
def validate_aadhaar_document(base64_data: str, file_type: str = "auto") -> Dict[str, Any]:
    """
    Validate and extract details from an Aadhaar card document provided as a Base64 string.
    
    Supports:
      - Formats: PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF (raw base64 or data URI)
      - OCR: RapidOCR (offline ONNX runtime)
      - SLM: Local GLiNER Small Language Model for NER extraction
      - Checksum: Verhoeff mathematical validation

    Returns structured dictionary with aadhaar_number, name, dob, gender, address, validity.
    """
    validator = get_validator()
    return validator.process_base64(base64_data, file_type=file_type)


@mcp.tool()
def validate_aadhaar_number(aadhaar_number: str) -> Dict[str, Any]:
    """
    Directly validates a 12-digit Aadhaar number string using the Verhoeff algorithm.
    Does not require document or OCR.
    """
    validator = get_validator()
    return validator.validate_number(aadhaar_number)


def main():
    parser = argparse.ArgumentParser(description="Aadhaar MCP Validator Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port for SSE transport (default: 8090)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local GLiNER model directory"
    )
    args = parser.parse_args()

    # Pre-warm the validator
    get_validator(model_path=args.model_path)

    if args.transport == "sse":
        print(f"Starting Aadhaar MCP Server on SSE port {args.port}...")
        try:
            mcp.run(transport="sse", host="0.0.0.0", port=args.port)
        except TypeError:
            if hasattr(mcp, "settings"):
                mcp.settings.port = args.port
            mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
