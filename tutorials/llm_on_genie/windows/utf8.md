# UTF-8 support on Windows

If you want to use Unicode characters in your prompts, please enable UTF-8 on
Windows:

* Open Control Panel -> Region -> Administrative -> Change system locale...
* Tick the "Beta: Use Unicode UTF-8 for worldwide language support"
* Restart Windows

It should now work. If there are still issues (e.g., question marks or
[mojibake[(https://en.wikipedia.org/wiki/Mojibake) instead of the expected
characters), try running this first in your PowerShell terminal:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
