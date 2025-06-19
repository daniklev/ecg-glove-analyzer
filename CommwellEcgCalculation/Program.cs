using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;

namespace CommwellEcgCalculation
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load your raw ECG data file
            string dataFilePath = "/Users/danik/Developer/ecg-glove-analyzer/data/220209015248248.ret";

            if (!File.Exists(dataFilePath))
            {
                Console.WriteLine($"Data file not found: {dataFilePath}");
                return;
            }

            try
            {
                // Read the raw byte data
                byte[] gloveData = File.ReadAllBytes(dataFilePath);
                Console.WriteLine($"Loaded file: {dataFilePath}");
                Console.WriteLine($"File size: {gloveData.Length} bytes");

                // Create CommCalc instance and process the data
                CommCalc calc = new CommCalc();
                Console.WriteLine($"CommCalc version: {calc.version}");

                Console.WriteLine("Processing ECG data...");
                int result = calc.gloveCalc(gloveData);


                Console.WriteLine($"gloveCalc returned: {result}");

                if (result == 0)
                {
                    Console.WriteLine("✓ ECG calculation successful!");

                    // Get the calculated values
                    int[] values = calc.getValues();
                    var waves = calc.getWaves();

                    if (values != null)
                    {
                        Console.WriteLine($"Calculated values ({values.Length}): [{string.Join(", ", values)}]");
                    }
                    else
                    {
                        Console.WriteLine("No calculated values returned");
                    }

                    if (waves != null)
                    {
                        Console.WriteLine($"Number of lead waves: {waves.Length}");
                        for (int i = 0; i < waves.Length; i++)
                        {
                            Console.WriteLine($"Lead {i + 1}: {waves[i].Count} samples");
                        }
                        SaveWavesToCsv(waves, dataFilePath);
                    }
                    else
                    {
                        Console.WriteLine("No wave data returned");
                    }
                }
                else
                {
                    Console.WriteLine("❌ ECG calculation failed");
                    Console.WriteLine("Reason: Insufficient RR intervals detected");
                    Console.WriteLine("This typically means:");
                    Console.WriteLine("  - Data format may not be compatible");
                    Console.WriteLine("  - ECG signal quality is too poor");
                    Console.WriteLine("  - File contains insufficient data");
                    Console.WriteLine("  - QRS detection algorithm couldn't find enough heartbeats");

                    // Try with other data files
                    Console.WriteLine("\nTrying other data files...");
                    TestOtherFiles();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing file: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        static void TestOtherFiles()
        {
            string[] testFiles = {
                "/Users/danik/Developer/ecg-glove-analyzer/data/220209012940240.ret",
                "/Users/danik/Developer/ecg-glove-analyzer/data/220209012940241.ret",
                "/Users/danik/Developer/ecg-glove-analyzer/CommwellEcgCalculation/20256912588244379.ret"
            };

            foreach (string testFile in testFiles)
            {
                if (File.Exists(testFile))
                {
                    Console.WriteLine($"\n--- Testing {Path.GetFileName(testFile)} ---");
                    try
                    {
                        byte[] data = File.ReadAllBytes(testFile);
                        Console.WriteLine($"File size: {data.Length} bytes");

                        CommCalc calc = new CommCalc();
                        int result = calc.gloveCalc(data);

                        if (result == 0)
                        {
                            Console.WriteLine("✓ SUCCESS with this file!");
                            int[] values = calc.getValues();
                            Console.WriteLine($"Values: [{string.Join(", ", values)}]");
                            return; // Stop on first success
                        }
                        else
                        {
                            Console.WriteLine("❌ Failed with this file too");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error with {testFile}: {ex.Message}");
                    }
                }
                else
                {
                    Console.WriteLine($"File not found: {testFile}");
                }
            }
        }

        static void SaveWavesToCsv(ConcurrentQueue<int>[] waves, string originalFilePath)
        {
            string csvPath = Path.ChangeExtension(originalFilePath+" c#", "csv");
            var sb = new StringBuilder();
            
            // Create header row with lead numbers
            for (int i = 0; i < waves.Length; i++)
            {
                sb.Append($"Lead {i + 1}");
                if (i < waves.Length - 1) sb.Append(',');
            }
            sb.AppendLine();

            // Find the maximum number of samples across all leads
            int maxSamples = waves.Max(w => w.Count);

            // Convert ConcurrentQueues to arrays for easier processing
            var waveArrays = waves.Select(q => q.ToArray()).ToArray();

            // Write sample values
            for (int i = 0; i < maxSamples; i++)
            {
                for (int lead = 0; lead < waves.Length; lead++)
                {
                    if (i < waveArrays[lead].Length)
                        sb.Append(waveArrays[lead][i]);
                    if (lead < waves.Length - 1) sb.Append(',');
                }
                sb.AppendLine();
            }

            File.WriteAllText(csvPath, sb.ToString());
            Console.WriteLine($"Wave data saved to: {csvPath}");
        }
    }
}