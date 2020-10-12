/* Below the code will change the input type */
alpha = document.getElementsByName("alpha_choice")[0]
alpha.type = "range";
beta = document.getElementsByName("beta_choice")[0]
beta.type = "range";

var alpha_demo = document.getElementById("alpha_demo");
var beta_demo = document.getElementById("beta_demo");

beta_demo.innerHTML = beta.value
alpha_demo.innerHTML = alpha.value


console.log(beta.value)

alpha.oninput = function alpha() {
    /* On input it will show the alpha value*/
    alpha_demo.innerHTML = this.value;
}

beta.oninput = function beta() {
    /* On input it will show the beta value*/
    beta_demo.innerHTML = this.value;

}