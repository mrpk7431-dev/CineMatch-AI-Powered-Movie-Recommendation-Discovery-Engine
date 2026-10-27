document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('movie-input');
    const list = document.getElementById('autocomplete-list');
    
    if (!input || !list) return;
    
    let currentFocus = -1;
    let timeout = null;
    
    input.addEventListener('input', function() {
        const val = this.value;
        closeAllLists();
        
        if (!val) { return false; }
        
        clearTimeout(timeout);
        timeout = setTimeout(async () => {
            currentFocus = -1;
            
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(val)}`);
                const movies = await response.json();
                
                if (movies.length > 0) {
                    list.style.display = 'block';
                    movies.forEach(movie => {
                        const item = document.createElement('div');
                        
                        // Make matching letters bold
                        const matchIndex = movie.toLowerCase().indexOf(val.toLowerCase());
                        if (matchIndex >= 0) {
                            item.innerHTML = movie.substring(0, matchIndex);
                            item.innerHTML += `<strong>${movie.substring(matchIndex, matchIndex + val.length)}</strong>`;
                            item.innerHTML += movie.substring(matchIndex + val.length);
                        } else {
                            item.innerHTML = movie;
                        }
                        
                        item.innerHTML += `<input type='hidden' value="${movie}">`;
                        
                        item.addEventListener('click', function(e) {
                            input.value = this.getElementsByTagName('input')[0].value;
                            closeAllLists();
                            document.getElementById('search-form').submit();
                        });
                        
                        list.appendChild(item);
                    });
                }
            } catch (err) {
                console.error("Error fetching autocomplete suggestions", err);
            }
        }, 300); // 300ms debounce
    });
    
    // Keyboard navigation
    input.addEventListener('keydown', function(e) {
        let x = document.getElementById("autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) { // Down
            currentFocus++;
            addActive(x);
        } else if (e.keyCode == 38) { // Up
            currentFocus--;
            addActive(x);
        } else if (e.keyCode == 13) { // Enter
            if (currentFocus > -1) {
                if (x) {
                    e.preventDefault();
                    x[currentFocus].click();
                }
            }
        }
    });
    
    function addActive(x) {
        if (!x) return false;
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        x[currentFocus].style.backgroundColor = "rgba(59, 130, 246, 0.2)";
    }
    
    function removeActive(x) {
        for (let i = 0; i < x.length; i++) {
            x[i].style.backgroundColor = "";
        }
    }
    
    function closeAllLists(elmnt) {
        if (elmnt != list && elmnt != input) {
            list.innerHTML = '';
            list.style.display = 'none';
        }
    }
    
    document.addEventListener("click", function(e) {
        closeAllLists(e.target);
    });
});
