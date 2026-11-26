using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using Microsoft.Win32;

static class Installer
{
    const string Title = "TEMPORAL Setup";

    static Stream Resource(string name)
    {
        return Assembly.GetExecutingAssembly().GetManifestResourceStream(name);
    }

    static string ReadText(string name)
    {
        using (var reader = new StreamReader(Resource(name))) return reader.ReadToEnd();
    }

    static void CopyTree(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (var file in Directory.GetFiles(from)) File.Copy(file, Path.Combine(to, Path.GetFileName(file)), true);
        foreach (var dir in Directory.GetDirectories(from)) CopyTree(dir, Path.Combine(to, Path.GetFileName(dir)));
    }

    [STAThread]
    static int Main(string[] args)
    {
        bool quiet = args.Any(a => a.Equals("/quiet", StringComparison.OrdinalIgnoreCase) || a.Equals("-quiet", StringComparison.OrdinalIgnoreCase));
        try
        {
            string[] info = ReadText("version.txt").Trim().Split('|');
            string version = info[0], publisher = info[1];
            if (!quiet && MessageBox.Show("Install TEMPORAL " + version + " for DaVinci Resolve?", Title, MessageBoxButtons.OKCancel, MessageBoxIcon.Question) != DialogResult.OK) return 2;
            while (Process.GetProcessesByName("Resolve").Length > 0)
            {
                if (quiet) throw new InvalidOperationException("DaVinci Resolve is running.");
                if (MessageBox.Show("Quit DaVinci Resolve, then press Retry.", Title, MessageBoxButtons.RetryCancel, MessageBoxIcon.Warning) != DialogResult.Retry) return 2;
            }

            string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Temporal");
            string plugins = Path.Combine(root, "OpenFX");
            string staging = Path.Combine(Path.GetTempPath(), "temporal-setup-" + Guid.NewGuid().ToString("N"));
            string zip = staging + ".zip";
            using (var input = Resource("payload.zip"))
            using (var output = File.Create(zip)) input.CopyTo(output);
            ZipFile.ExtractToDirectory(zip, staging);
            Directory.CreateDirectory(plugins);
            foreach (var bundle in Directory.GetDirectories(staging, "*.ofx.bundle"))
            {
                string target = Path.Combine(plugins, Path.GetFileName(bundle));
                if (Directory.Exists(target)) Directory.Delete(target, true);
                CopyTree(bundle, target);
            }
            Directory.Delete(staging, true);
            File.Delete(zip);
            File.WriteAllText(Path.Combine(root, "uninstall.ps1"), ReadText("uninstall.ps1"));

            string current = Environment.GetEnvironmentVariable("OFX_PLUGIN_PATH", EnvironmentVariableTarget.User) ?? "";
            var parts = current.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries).ToList();
            if (!parts.Any(p => string.Equals(p.TrimEnd('\\'), plugins, StringComparison.OrdinalIgnoreCase)))
            {
                parts.Add(plugins);
                Environment.SetEnvironmentVariable("OFX_PLUGIN_PATH", string.Join(";", parts), EnvironmentVariableTarget.User);
            }

            using (var key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\TEMPORAL"))
            {
                key.SetValue("DisplayName", "TEMPORAL OFX");
                key.SetValue("DisplayVersion", version);
                key.SetValue("Publisher", publisher);
                key.SetValue("InstallLocation", root);
                key.SetValue("DisplayIcon", Path.Combine(plugins, @"TEMPORAL.ofx.bundle\Contents\Resources\temporal.ico"));
                key.SetValue("UninstallString", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"" + Path.Combine(root, "uninstall.ps1") + "\"");
                key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
            }
            if (!quiet) MessageBox.Show("TEMPORAL " + version + " is installed. Start DaVinci Resolve to use it.", Title, MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 0;
        }
        catch (Exception ex)
        {
            if (!quiet) MessageBox.Show("Installation failed: " + ex.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            else Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }
}
