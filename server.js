const express = require('express');
const { spawn } = require('child_process');
const readline = require('node:readline');
const SSE = require('express-sse');

var serverSentEvents = new SSE();

async function startPoseDetectorProcess() {
    const subprocess = spawn(
        "python",
        ["posedetector.py"],
        // When the subprocess crashes, use this line to debug the output directly.
        // { stdio: ['pipe', process.stdout, process.sterr] }
    );

    // Handle errors during subprocess execution
    subprocess.on('error', (err) => {
        console.error(`Failed to start subprocess: ${err}`);
    });

    // Handle subprocess exit
    subprocess.on('close', (code) => {
        console.log(`Subprocess exited with code ${code}`);
    });

    // Read output line by line - stdout.
    const rl = readline.createInterface({
        input: subprocess.stdout,
        crlfDelay: Infinity, // To handle both \r\n and \n line endings
    });

    rl.on('line', (line) => {
        // console.log(`Line from subprocess: ${line}`);
        const decoded = JSON.parse(line);
        serverSentEvents.send(decoded.content, decoded.message_type);
    });

    // Handle errors while reading lines
    rl.on('error', (err) => {
        console.error(`Error reading from subprocess: ${err}`);
    });

    // Read output line by line - stderr.
    const rlStdErr = readline.createInterface({
        input: subprocess.stderr,
        crlfDelay: Infinity, // To handle both \r\n and \n line endings
    });

    rlStdErr.on('line', (line) => {
        console.warn(`Stderr from subprocess: ${line}`);
    });

    // Handle errors while reading lines
    rlStdErr.on('error', (err) => {
        console.warn(`Error reading from subprocess: ${err}`);
    });
}

const app = express();
const port = 3000;

app.get('/stream', serverSentEvents.init);

app.get('/foo', (req, res) => {
    res.send('Hello World!')
})

app.use(express.static('public'));

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`)
});

startPoseDetectorProcess();