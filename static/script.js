/* =========================================================
   [Part 1] 전역 함수 (HTML 태그에서 직접 호출하는 함수들)
   - 스케일링, 체크박스 제어, 팝업 열기 등
   ========================================================= */

// 1. 화면 스케일링 (1440x900 비율 유지)
const BASE_WIDTH = 1440;
const BASE_HEIGHT = 900;

function applyScale() {
  const wrapper = document.getElementById("scale-wrapper");
  if (!wrapper) return; // 팝업창 등 scale-wrapper가 없는 경우 중단

  const scaleX = window.innerWidth / BASE_WIDTH;
  const scaleY = window.innerHeight / BASE_HEIGHT;
  const scale = Math.min(scaleX, scaleY);
  
  wrapper.style.transform = `scale(${scale})`;
}

// 창 크기 변경 시 자동 적용
window.addEventListener("resize", applyScale);


// 2. 즐겨찾기: 모든 체크박스 제어 (개별 체크박스 -> 전체)
function toggleAll(checked) {
  // tbody 안의 모든 체크박스 제어
  const checkboxes = document.querySelectorAll('.fav-table tbody input[type="checkbox"]');
  checkboxes.forEach(cb => {
    cb.checked = checked;
  });

  // 상단 헤더 체크박스 상태 동기화
  const headCheck = document.getElementById('check-all');
  if (headCheck) headCheck.checked = checked;
}


// 3. 즐겨찾기: "전체 선택" 버튼 클릭 시 실행
function toggleAllByButton() {
  const headCheck = document.getElementById('check-all');
  if (headCheck) {
    // 현재 헤더 체크박스 상태를 반전시켜서 적용
    headCheck.checked = !headCheck.checked;
    toggleAll(headCheck.checked);
  }
}

// 4. 팝업창 열기 (chat.html 내 자세히보기 버튼 등에서 사용)

function openPopup(url) {
  window.open(
    url,
    "answerDetailPopup",
    "width=900,height=700,scrollbars=yes"
  );
}

// 5. 쿼리 보기 토글 (채팅 화면)
function toggleQueryBar() {
  const content = document.getElementById("queryContent");
  const icon = document.getElementById("queryToggleIcon");
  if (!content || !icon) return;

  const isHidden = content.style.display === "none";
  content.style.display = isHidden ? "block" : "none";
  icon.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
}

document.querySelectorAll('.status-toggle').forEach(toggle => {
  const label = toggle.closest('.switch').querySelector('.label-text');

  function sync() {
    // ✅ 기존 기능: 텍스트 변경
    label.textContent = toggle.checked ? '활성화' : '비활성화';
  }

  sync(); // 초기 상태 반영

  toggle.addEventListener('change', function () {
    sync(); // 기존 동작 유지

    // ✅ 추가 기능: Flask로 즉시 전송
    fetch('/manage_role', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_seq: toggle.dataset.userSeq,
        is_active: toggle.checked ? 'Y' : 'N'
      })
    })
    .then(res => {
      if (!res.ok) throw new Error('서버 오류');
    })
    .catch(err => {
      // ❗ 실패 시 롤백 (UX 중요)
      toggle.checked = !toggle.checked;
      sync();
      alert('상태 변경에 실패했습니다.');
    });
  });
});



/* =========================================================
   [Part 2] DOM 로드 후 실행 (화면 전환 및 UI 이벤트 리스너)
   ========================================================= */
document.addEventListener("DOMContentLoaded", () => {
  
  // 1. 초기 스케일 적용
  applyScale();


  /**
   * SPA 화면 전환 로직
   * - 의료 도메인 환경의 빠른 데이터 응답성을 위해 전체 리로드 없이 섹션만 교체 [cite: 326, 329]
   */

  window.showScreen = function(screenName) {
    // 1. 모든 화면 섹션(.center-area) 숨기기
    document.querySelectorAll('.center-area').forEach(screen => {
      screen.style.display = 'none';
    });

    // 2. 선택된 화면만 보이기
    const target = document.getElementById('screen-' + screenName);
    if (target) {
      // CSS에 정의된 flex 레이아웃 유지
      target.style.display = 'flex';
    }

    // 3. 사이드바 UI 상태 및 레이아웃 모드 업데이트
    updateSidebar(screenName);
    updateMainLayout(screenName);
  };

  /**
   * 사이드바 활성화 상태 업데이트
   */
  function updateSidebar(screenName) {
    const allMenuItems = document.querySelectorAll('.menu-item, .recent-item');
    allMenuItems.forEach(item => item.classList.remove('active'));

    const btnHome = document.getElementById('btn-home');
    const btnFav = document.getElementById('btn-fav');

    if (screenName === 'fav') {
      if (btnFav) btnFav.classList.add('active');
    } else {
      // 홈(home)과 채팅(chat) 화면은 '새로운 채팅' 메뉴를 활성화 [cite: 443]
      if (btnHome) btnHome.classList.add('active');
    }
  }

  /**
   * 화면별 특수 레이아웃 처리 (즐겨찾기 페이지 등 스크롤 허용 여부)
   */
  function updateMainLayout(screenName) {
    const mainContent = document.querySelector('.main-content');
    if (screenName === 'fav') {
      mainContent.classList.add('fav-mode'); // CSS에서 정의한 즐겨찾기용 패딩/스크롤 적용
    } else {
      mainContent.classList.remove('fav-mode');
    }
  }

  // 초기 로드 시 홈 화면 설정
  document.addEventListener('DOMContentLoaded', () => {
    showScreen('home');
  });

  // 상세 보기 팝업 (임시)
  function openPopup() {
    alert("상세 데이터를 Oracle DB에서 조회합니다. [cite: 423]");
  }


  // ============================
  // [B] 이벤트 리스너 연결
  // ============================

  // 1) 사이드바 메뉴 클릭
  const btnHome = document.getElementById('btn-home');
  const btnFav = document.getElementById('btn-fav');

  if(btnHome) btnHome.addEventListener('click', () => showScreen('home'));
  if(btnFav) btnFav.addEventListener('click', () => showScreen('fav'));

  // 2) 사이드바 최근 기록 클릭 -> 채팅 화면으로
  const recentItems = document.querySelectorAll(".recent-section .recent-item");
  recentItems.forEach((div) => {
    div.addEventListener("click", () => {
      showScreen('chat');
      // 최근 기록 항목 자체 하이라이트 (선택 사항)
      document.querySelectorAll(".menu-item").forEach(i => i.classList.remove("active"));
      div.classList.add("active");
    });
  });

  // 3) 검색 동작 (버튼 클릭 or 엔터) -> 채팅 화면으로
  const searchBtns = document.querySelectorAll(".search-button");
  const searchInputs = document.querySelectorAll(".search-box input");

  if (searchBtns.length > 0) {
    searchBtns.forEach(btn => btn.addEventListener("click", () => showScreen('chat')));
  }
  if (searchInputs.length > 0) {
    searchInputs.forEach(input => {
      input.addEventListener("keypress", (e) => {
        if (e.key === "Enter") showScreen('chat');
      });
    });
  }

  // 4) 추천 칩 클릭 -> 채팅 화면으로
  const chips = document.querySelectorAll(".suggestion-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => showScreen('chat'));
  });


  

});

document.addEventListener("DOMContentLoaded", () => {

  /* =================================================
     [1] 헤더 버튼 기능 구현 (창닫기, 재실행, 다운로드)
     ================================================= */
  
  // 모든 헤더 버튼을 가져옵니다.
  const headerBtns = document.querySelectorAll(".header-btn");

  headerBtns.forEach(btn => {
    const btnText = btn.innerText.trim(); // 버튼 안의 텍스트("재실행", "다운로드" 등)

    // 1. 창닫기 버튼 (클래스 .close로 식별)
    if (btn.classList.contains("close")) {
      btn.addEventListener("click", () => {
        window.close(); // 팝업창 닫기
      });
    }
    
    // 2. 재실행 버튼 (텍스트로 식별)
    else if (btnText.includes("재실행")) {
      btn.addEventListener("click", () => {
        // 페이지를 새로고침하여 쿼리 재실행 효과를 냄
        location.reload(); 
      });
    }

    // 3. 다운로드 버튼 (텍스트로 식별, 선택사항)
    else if (btnText.includes("다운로드")) {
      btn.addEventListener("click", () => {

        alert(
          "결과 파일을 다운로드합니다.\n" +
          "최대 다운로드가 가능한 행 수는 100,000행입니다."
        );

        const pathParts = window.location.pathname.split('/');
        const answerId = pathParts[2];

        window.location.href = `/popup/${answerId}/download`;

      });
    }
  });


  /* =================================================
     [2] 기존 기능 유지 (스케일링, 쿼리 토글 등)
     ================================================= */
  
  // 쿼리 토글 기능 (HTML onclick에서도 호출 가능하도록 window 객체에 할당)
  window.toggleQueryBar = function() {
    const content = document.getElementById("queryContent");
    const icon = document.getElementById("queryToggleIcon");
    
    if (!content || !icon) return;

    const isHidden = content.style.display === "none";
    if (isHidden) {
      content.style.display = "block";
      icon.style.transform = "rotate(180deg)";
    } else {
      content.style.display = "none";
      icon.style.transform = "rotate(0deg)";
    }
  };

  // 즐겨찾기: 입력값 있을 때만 버튼 활성화 + 클릭 시 꽉 찬 별
  const favToggle = document.querySelector(".favorite-toggle");
  const favInput = document.getElementById("favoriteName");
  const favClear = document.querySelector(".favorite-clear");

  function syncFavoriteButtonState() {
    if (!favToggle || !favInput) return;
    const hasValue = favInput.value.trim().length > 0;
    favToggle.disabled = !hasValue;
  }

  if (favInput) {
    favInput.addEventListener("input", syncFavoriteButtonState);
  }

  if (favToggle) {
    favToggle.addEventListener("click", function() {
      if (favInput && favInput.value.trim().length > 0) {
        this.classList.toggle("is-filled");
      }
    });
  }

  if (favClear) {
    favClear.addEventListener("click", function() {
      if (!favInput) return;
      favInput.value = "";
      if (favToggle) favToggle.classList.remove("is-filled");
      syncFavoriteButtonState();
      favInput.focus();
    });
  }

  // 초기 상태 동기화
  syncFavoriteButtonState();

});


document.addEventListener('DOMContentLoaded', () => {
  const currentPath = window.location.pathname;

  if (currentPath === '/system_log') {
    document
      .getElementById('menu-system-log')
      ?.classList.add('active');
  }
});

document.addEventListener('DOMContentLoaded', () => {
  const currentPath = window.location.pathname;

  if (currentPath === '/manage_role') {
    document
      .getElementById('manage_role')
      ?.classList.add('active');
  }
});

document.addEventListener('DOMContentLoaded', () => {
  const currentPath = window.location.pathname;

  if (currentPath === '/favorite') {
    document
      .getElementById('favorite')
      ?.classList.add('active');
  }
});