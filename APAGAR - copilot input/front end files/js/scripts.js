window.onload = function (e) {
    if (document.getElementById("submitCalculationBtn")) {
        document.getElementById("submitCalculationBtn").onclick = function () {
            document.getElementById("submitCalculationAnimate").classList.remove("hidden");

            this.classList.add("opacity-50");
        }
    }

    if (document.getElementById("submitCalculationAgainBtn")) {
        document.getElementById("submitCalculationAgainBtn").onclick = function () {
            document.getElementById("submitCalculationAgainAnimate").classList.remove("hidden");
            this.classList.add("opacity-50");
        }

        if (document.getElementById("TrainBtn")) {
            document.getElementById("TrainBtn").onclick = function () {
                document.getElementById("trainAnimate").classList.remove("hidden");
                this.classList.add("opacity-50");
            }
        }

    }

    var queryString = window.location.search;
    var urlParams = new URLSearchParams(queryString);
    if (urlParams.get("query")) {
        validateStep(2);
        document.getElementById("query").value = urlParams.get("query");
        document.getElementById("country").value = urlParams.get("query");
    }
    if (urlParams.get("country")) {
        validateStep(2);
        document.getElementById("query").value = urlParams.get("country");
        document.getElementById("country").value = urlParams.get("country");
    }
    if (window.location.href.indexOf("calculus") !== -1) {
        validateStep(3);
    }
}

function validateStep(stepNumber) {
    document.getElementById("step" + stepNumber).classList.remove("border-gray-200");
    document.getElementById("step" + stepNumber).classList.add("border-green-400");
    document.getElementById("step" + stepNumber).firstElementChild.classList.remove("text-gray-400");
    document.getElementById("step" + stepNumber).firstElementChild.classList.add("text-green-400");
}



// script.js
