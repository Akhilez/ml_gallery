export default class Transporter {
    constructor(project_id, call_back, trace_id) {
        this.project_id = project_id;
        this.call_back = call_back;
        this.trace_id = trace_id;
    }

    init() {
        /*
        Must be called when component is mounted.
        Or
        Must be called when a socket is opened
        */
    }

    send(data) {
        throw new Error("Send method must be implemented.");
    }

    received(data) {
        if (this.trace_id == null)
            this.trace_id = data.trace_id;
        else if (data.trace_id !== this.trace_id) {
            console.log("Trace ID does not match.");
            return;
        }
        this.call_back(data);
    }

}