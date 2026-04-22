"""Custom AIO Sandbox backend for DeepAgents.

Wraps the agent-infra/sandbox SDK to provide Shell, File, and execution
capabilities inside an isolated Docker container.

Usage:
    1. Start the sandbox container:
       docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 \
           ghcr.io/agent-infra/sandbox:latest

    2. Use in your agent:
       from sandbox_backend import AIOSandboxBackend
       backend = AIOSandboxBackend(base_url="http://localhost:8080")
       agent = create_deep_agent(model=model, backend=backend)

Requires: pip install agent-sandbox
"""

import os
import uuid
from pathlib import PurePosixPath

from deepagents.backends.protocol import (
    BackendProtocol,
    SandboxBackendProtocol,
    ReadResult,
    WriteResult,
    EditResult,
    LsResult,
    GlobResult,
    GrepResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from agent_sandbox import Sandbox


class AIOSandboxBackend(BaseSandbox):
    """DeepAgents backend that delegates to an AIO Sandbox Docker container.

    Inherits from BaseSandbox which provides file operations (read, write, edit,
    glob, grep, ls) on top of a shell execute method. We implement execute()
    and file upload/download to complete the interface.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize connection to AIO Sandbox.

        Args:
            base_url: URL of the running AIO Sandbox container.
        """
        self.client = Sandbox(base_url=base_url)
        self.base_url = base_url
        self._id = f"aio-sandbox-{uuid.uuid4().hex[:8]}"

        # Get sandbox context (home directory, etc.)
        try:
            ctx = self.client.sandbox.get_context()
            self.home_dir = ctx.home_dir
        except Exception:
            self.home_dir = "/home/gem"

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox instance."""
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to run.
            timeout: Max seconds to wait for command completion.

        Returns:
            ExecuteResponse with output and exit_code.
        """
        try:
            res = self.client.shell.exec_command(command=command)
            return ExecuteResponse(
                output=res.data.output if res.data else "",
                exit_code=0,
            )
        except Exception as e:
            return ExecuteResponse(
                output=str(e),
                exit_code=1,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the sandbox.

        Args:
            files: List of (sandbox_path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse for each file.
        """
        responses = []
        for path, content in files:
            try:
                # Ensure parent directory exists
                parent = str(PurePosixPath(path).parent)
                if parent != "/":
                    self.client.shell.exec_command(command=f"mkdir -p {parent}")
                # Write content (decode bytes to string for text files)
                text = content.decode("utf-8", errors="replace")
                self.client.file.write_file(file=path, content=text)
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Args:
            paths: List of file paths in the sandbox.

        Returns:
            List of FileDownloadResponse with content bytes.
        """
        responses = []
        for path in paths:
            try:
                res = self.client.file.read_file(file=path)
                content = (res.data.content if res.data else "").encode("utf-8")
                responses.append(FileDownloadResponse(path=path, content=content))
            except Exception:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
        return responses

    # -----------------------------------------------------------------------
    # Convenience methods (not part of BaseSandbox interface)
    # -----------------------------------------------------------------------

    def upload(self, local_path: str, sandbox_path: str) -> None:
        """Upload a local file to the sandbox (convenience method).

        Args:
            local_path: Path to the local file.
            sandbox_path: Destination path in the sandbox.
        """
        with open(local_path, "rb") as f:
            content = f.read()
        self.upload_files([(sandbox_path, content)])

    def download(self, sandbox_path: str, local_path: str) -> None:
        """Download a file from the sandbox to local filesystem (convenience method).

        Args:
            sandbox_path: Path to the file in the sandbox.
            local_path: Destination path on local filesystem.
        """
        responses = self.download_files([sandbox_path])
        if responses and responses[0].content is not None:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(responses[0].content)
        else:
            raise FileNotFoundError(f"Could not download {sandbox_path}")

    def setup_environment(self, packages: list[str] | None = None) -> str:
        """Install Python packages in the sandbox.

        Args:
            packages: List of pip packages to install.

        Returns:
            Install output.
        """
        if not packages:
            packages = ["pandas", "numpy", "scikit-learn"]
        cmd = f"pip install {' '.join(packages)}"
        result = self.execute(cmd, timeout=300)
        return result.output

    def upload_data_files(self, file_paths: dict[str, str]) -> None:
        """Upload multiple local files to sandbox.

        Args:
            file_paths: Dict of {sandbox_path: local_path}.
        """
        # Ensure data directory exists
        self.execute(f"mkdir -p {self.home_dir}/data")

        for sandbox_path, local_path in file_paths.items():
            self.upload(local_path, sandbox_path)
