// GWASMind.exe: starts the bundled Python runtime with the GWASMind web service.
//
//   GWASMind.exe [server options]     local web interface (opens the browser)
//   GWASMind.exe --no-browser [...]   local web interface without opening a browser
//   GWASMind.exe cli <command> [...]  command-line interface (gwasmind.cli)
//
// Server options are passed to "python -m gwasmind.web.server", for example
// --default-mode frozen-safe --frozen-data-root D:\opentargets\25.09
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Microsoft.Win32;

internal static class GWASMindLauncher
{
    private static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, "runtime", "python.exe");
        if (!File.Exists(python))
        {
            Console.Error.WriteLine("The bundled Python runtime was not found: " + python);
            Console.Error.WriteLine("Reinstall GWASMind to repair the installation.");
            return 2;
        }

        var arguments = new List<string>();
        bool cli = args.Length > 0 && args[0] == "cli";
        if (cli)
        {
            arguments.Add("-m");
            arguments.Add("gwasmind.cli");
            for (int i = 1; i < args.Length; i++) arguments.Add(args[i]);
        }
        else
        {
            Console.WriteLine("GWASMind - local service. Close this window or press Ctrl+C to stop it.");
            WarnIfLongPathsDisabled();
            arguments.Add("-m");
            arguments.Add("gwasmind.web.server");
            bool openBrowser = true;
            foreach (string value in args)
            {
                if (value == "--no-browser") openBrowser = false;
                else arguments.Add(value);
            }
            if (openBrowser) arguments.Add("--open-browser");
        }

        var start = new ProcessStartInfo(python, Join(arguments))
        {
            UseShellExecute = false,
            WorkingDirectory = root
        };
        // Isolate the bundled runtime from any other Python on this computer.
        start.EnvironmentVariables.Remove("PYTHONHOME");
        start.EnvironmentVariables.Remove("PYTHONPATH");
        start.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        start.EnvironmentVariables["PYTHONUTF8"] = "1";

        // Ctrl+C reaches both processes; the service shuts down cleanly, and
        // this launcher waits for it instead of exiting first.
        Console.CancelKeyPress += (sender, e) => { e.Cancel = true; };
        using (Process child = Process.Start(start))
        {
            child.WaitForExit();
            return child.ExitCode;
        }
    }

    private static void WarnIfLongPathsDisabled()
    {
        object value = null;
        try
        {
            using (RegistryKey key = Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Control\FileSystem"))
            {
                if (key != null) value = key.GetValue("LongPathsEnabled");
            }
        }
        catch (Exception)
        {
            return;
        }
        if (!(value is int) || (int)value != 1)
        {
            Console.WriteLine("WARNING: Windows long-path support is disabled. Saved analyses can exceed 260");
            Console.WriteLine("characters; ask an administrator to enable LongPathsEnabled (see README).");
        }
    }

    // Quote arguments for the Windows command line (CommandLineToArgvW rules).
    private static string Join(IEnumerable<string> values)
    {
        var line = new StringBuilder();
        foreach (string value in values)
        {
            if (line.Length > 0) line.Append(' ');
            if (value.Length > 0 && value.IndexOfAny(new[] { ' ', '\t', '"' }) < 0)
            {
                line.Append(value);
                continue;
            }
            line.Append('"');
            int backslashes = 0;
            foreach (char character in value)
            {
                if (character == '\\') { backslashes++; continue; }
                if (character == '"') line.Append('\\', backslashes * 2 + 1);
                else line.Append('\\', backslashes);
                backslashes = 0;
                line.Append(character);
            }
            line.Append('\\', backslashes * 2);
            line.Append('"');
        }
        return line.ToString();
    }
}
